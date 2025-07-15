#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright author: fumo2024

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # 获取实验名（如有）
    experiment_name = getattr(opt, 'experiment_name', None)
    tb_prefix = f"{experiment_name}/" if experiment_name else ""
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    total_train_time = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    prev_xyz = None
    prev_cov = None
    # 信息增益视角选取相关变量
    import numpy as np
    residual_history = {}  # {viewpoint: [loss1, loss2, ...]}
    trained_view_features = []  # [phi(v1), phi(v2), ...]
    def get_view_feature(viewpoint):
        # 位置、方向、焦距
        pos = getattr(viewpoint, 'position', None)
        dir = getattr(viewpoint, 'direction', None)
        focal = np.array([getattr(viewpoint, 'focal_length', 1.0)])
        if pos is None:
            pos = np.array([0,0,0])
        if dir is None:
            dir = np.array([0,0,1])
        return np.concatenate([np.array(pos).flatten(), np.array(dir).flatten(), focal])
    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    def compute_g_res(viewpoint):
        losses = residual_history.get(viewpoint, [])
        if len(losses) < 1:
            return 0.0
        return float(np.mean(losses[-20:]))
    def compute_g_nov(viewpoint):
        phi_v = get_view_feature(viewpoint)
        if not trained_view_features:
            return 1.0
        sims = [cosine_similarity(phi_v, phi_vp) for phi_vp in trained_view_features]
        return 1.0 - max(sims)
    def select_viewpoint(candidate_viewpoints, alpha=0.5):
        g_res_list = [compute_g_res(v) for v in candidate_viewpoints]
        g_nov_list = [compute_g_nov(v) for v in candidate_viewpoints]
        # min-max归一化
        g_res_min, g_res_max = min(g_res_list), max(g_res_list)
        g_nov_min, g_nov_max = min(g_nov_list), max(g_nov_list)
        g_res_norm = [(x - g_res_min) / (g_res_max - g_res_min + 1e-8) for x in g_res_list]
        g_nov_norm = [(x - g_nov_min) / (g_nov_max - g_nov_min + 1e-8) for x in g_nov_list]
        scores = [alpha * r + (1 - alpha) * n for r, n in zip(g_res_norm, g_nov_norm)]
        best_idx = int(np.argmax(scores))
        return candidate_viewpoints[best_idx]
    def weighted_sample_viewpoint(candidate_viewpoints, alpha=0.5):
        g_res_list = [compute_g_res(v) for v in candidate_viewpoints]
        g_nov_list = [compute_g_nov(v) for v in candidate_viewpoints]
        g_res_min, g_res_max = min(g_res_list), max(g_res_list)
        g_nov_min, g_nov_max = min(g_nov_list), max(g_nov_list)
        g_res_norm = [(x - g_res_min) / (g_res_max - g_res_min + 1e-8) for x in g_res_list]
        g_nov_norm = [(x - g_nov_min) / (g_nov_max - g_nov_min + 1e-8) for x in g_nov_list]
        scores = [alpha * r + (1 - alpha) * n for r, n in zip(g_res_norm, g_nov_norm)]
        scores_np = np.array(scores)
        probs = scores_np / (scores_np.sum() + 1e-8)
        idx = np.random.choice(len(candidate_viewpoints), p=probs)
        return candidate_viewpoints[idx]
    # 低分辨率预热机制相关变量
    enable_warmup = getattr(opt, 'enable_warmup', False)
    warmup_stage = 0  # 0: 1/4分辨率, 1: 1/2分辨率, 2: 全分辨率
    warmup_tau = getattr(opt, 'warmup_tau', 0.005)  # ΔG阈值，默认0.5%
    warmup_count = 0  # 连续ΔG低于阈值计数
    warmup_count_limit = getattr(opt, 'warmup_count_limit', 100)
    warmup_sh_degree_trigger = getattr(opt, 'warmup_sh_degree_trigger', 2)
    warmup_sh_iter_trigger = getattr(opt, 'warmup_sh_iter_trigger', 1000)
    # 分辨率缩放因子
    warmup_scale = [0.25, 0.5, 1.0]
    # 兜底机制：最大预热迭代数
    warmup_max_iter_0 = getattr(opt, 'warmup_max_iter_0', 5000)  # stage 0最大迭代
    warmup_max_iter_1 = getattr(opt, 'warmup_max_iter_1', 5000)  # stage 1最大迭代
    enable_info_gain = getattr(opt, 'enable_info_gain', False)
    # 信息增益选取相关参数
    info_gain_start_iter = getattr(opt, 'info_gain_start_iter', 4000)
    info_gain_alternate_period = getattr(opt, 'info_gain_alternate_period', 2000)
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 视角选取逻辑（前期均匀，后期交替）
        if iteration < info_gain_start_iter:
            # 均匀抽样（不重复）
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            # 交替抽样
            use_info_gain = ((iteration - info_gain_start_iter) // info_gain_alternate_period) % 2 == 1
            if use_info_gain and enable_info_gain:
                # 优先级抽样始终用全量训练视角集合，允许重复
                candidate_viewpoints = scene.getTrainCameras()
                if len(candidate_viewpoints) > 10:
                    candidate_viewpoints = [candidate_viewpoints[i] for i in np.random.choice(len(candidate_viewpoints), min(10, len(candidate_viewpoints)), replace=False)]
                viewpoint_cam = weighted_sample_viewpoint(candidate_viewpoints, alpha=0.5)
            else:
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # 低分辨率预热机制：动态调整图像分辨率
        if enable_warmup:
            cur_scale = warmup_scale[warmup_stage]
        else:
            cur_scale = 1.0
        # 保存原始分辨率（只需一次）
        if not hasattr(viewpoint_cam, '_orig_height'):
            viewpoint_cam._orig_height = int(viewpoint_cam.image_height)
        if not hasattr(viewpoint_cam, '_orig_width'):
            viewpoint_cam._orig_width = int(viewpoint_cam.image_width)
        # 动态调整分辨率
        viewpoint_cam.image_height = int(viewpoint_cam._orig_height * cur_scale)
        viewpoint_cam.image_width = int(viewpoint_cam._orig_width * cur_scale)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # 渲染后恢复分辨率，避免影响后续
        viewpoint_cam.image_height = viewpoint_cam._orig_height
        viewpoint_cam.image_width = viewpoint_cam._orig_width

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        # 信息增益统计
        # 残差增益历史
        if viewpoint_cam not in residual_history:
            residual_history[viewpoint_cam] = []
        residual_history[viewpoint_cam].append(Ll1.item())
        # 训练视角特征
        trained_view_features.append(get_view_feature(viewpoint_cam))

        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            # record total iteration time
            torch.cuda.synchronize()
            iter_time = iter_start.elapsed_time(iter_end)
            total_train_time += iter_time
            if tb_writer:
                tb_writer.add_scalar(f'{tb_prefix}train/total_time', total_train_time, iteration)
            # 记录高斯点数、位置变化率、协方差变化率
            gauss_count = gaussians.get_xyz.shape[0] if hasattr(gaussians, 'get_xyz') else None
            xyz_change = None
            cov_change = None
            cur_xyz = gaussians.get_xyz if hasattr(gaussians, 'get_xyz') else None
            cur_cov = gaussians.get_covariance() if hasattr(gaussians, 'get_covariance') else None
            delta_g = None
            if prev_xyz is not None and cur_xyz is not None:
                if prev_xyz.shape == cur_xyz.shape:
                    xyz_change = torch.norm(cur_xyz - prev_xyz)
                    # ΔG变化率（百分比），分母更鲁棒
                    delta_g = torch.norm(cur_xyz - prev_xyz) / (torch.norm(prev_xyz) + torch.norm(cur_xyz) + 1e-8)
                else:
                    xyz_change = None  # 点数不一致时跳过
            if prev_cov is not None and cur_cov is not None:
                if prev_cov.shape == cur_cov.shape:
                    cov_change = torch.norm(cur_cov - prev_cov)
                else:
                    cov_change = None
            # 低分辨率预热机制阶段切换逻辑
            if enable_warmup:
                # stage 0: ΔG收敛或迭代数兜底
                if warmup_stage == 0:
                    if delta_g is not None and delta_g < warmup_tau:
                        warmup_count += 1
                    else:
                        warmup_count = 0
                    if warmup_count >= warmup_count_limit:
                        warmup_stage = 1  # 进入1/2分辨率
                        print(f"[Warmup] Geometry converged at iter {iteration}, switch to 1/2 resolution.")
                    elif iteration - first_iter > warmup_max_iter_0:
                        warmup_stage = 1
                        print(f"[Warmup] Max iter reached at iter {iteration}, switch to 1/2 resolution.")
                # stage 1: SH degree触发或迭代数兜底
                elif warmup_stage == 1:
                    sh_degree = getattr(gaussians, 'sh_degree', 0)
                    if sh_degree >= warmup_sh_degree_trigger and iteration >= warmup_sh_iter_trigger:
                        warmup_stage = 2
                        print(f"[Warmup] SH optimization started at iter {iteration}, switch to full resolution.")
                    elif iteration - first_iter > warmup_max_iter_1:
                        warmup_stage = 2
                        print(f"[Warmup] Max iter reached at iter {iteration}, switch to full resolution.")
            if tb_writer:
                if gauss_count is not None:
                    tb_writer.add_scalar(f'{tb_prefix}gaussians/count', gauss_count, iteration)
                if xyz_change is not None:
                    tb_writer.add_scalar(f'{tb_prefix}gaussians/xyz_change', xyz_change.item(), iteration)
                if cov_change is not None:
                    tb_writer.add_scalar(f'{tb_prefix}gaussians/cov_change', cov_change.item(), iteration)
                if delta_g is not None:
                    tb_writer.add_scalar(f'{tb_prefix}gaussians/delta_g', float(delta_g.item()), iteration)
                else:
                    tb_writer.add_scalar(f'{tb_prefix}gaussians/delta_g', 0.0, iteration)
                if enable_warmup:
                    tb_writer.add_scalar(f'{tb_prefix}warmup/stage', warmup_stage, iteration)
                    tb_writer.add_scalar(f'{tb_prefix}warmup/count', warmup_count, iteration)
                    tb_writer.add_scalar(f'{tb_prefix}warmup/tau', warmup_tau, iteration)
            prev_xyz = cur_xyz.detach().clone() if cur_xyz is not None else None
            prev_cov = cur_cov.detach().clone() if cur_cov is not None else None

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # TensorBoard 日志目录统一放在 output/tensorboard_logs/实验名 或 output/tensorboard_logs/模型名
    tb_writer = None
    if TENSORBOARD_FOUND:
        import time
        if getattr(args, 'experiment_name', None):
            tb_log_dir = os.path.join("output", "tensorboard_logs", args.experiment_name)
        else:
            tb_log_dir = os.path.join("output", "tensorboard_logs", os.path.basename(args.model_path))
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard log dir: {tb_log_dir}")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        # 支持实验分组
        experiment_name = getattr(scene, 'experiment_name', None)
        tb_prefix = f"{experiment_name}/" if experiment_name else ""
        tb_writer.add_scalar(f'{tb_prefix}train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{tb_prefix}train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{tb_prefix}iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        experiment_name = getattr(scene, 'experiment_name', None)
        tb_prefix = f"{experiment_name}/" if experiment_name else ""
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f'{tb_prefix}{config["name"]}_view_{viewpoint.image_name}/render', image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{tb_prefix}{config["name"]}_view_{viewpoint.image_name}/ground_truth', gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(f'{tb_prefix}{config["name"]}/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{tb_prefix}{config["name"]}/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f'{tb_prefix}scene/opacity_histogram', scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{tb_prefix}total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--no_gui', action='store_true', default=False)
    parser.add_argument('--enable_warmup', action='store_true', default=False, help='是否启用低分辨率预热机制')
    parser.add_argument('--enable_info_gain', action='store_true', default=False, help='是否启用信息增益视角选取')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称，用于TensorBoard分组')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # 低分辨率预热机制相关超参数
    if args.enable_warmup:
        if not hasattr(args, 'warmup_tau'):
            args.warmup_tau = 0.005  # ΔG阈值，默认0.5%
        if not hasattr(args, 'warmup_count_limit'):
            args.warmup_count_limit = 100
        if not hasattr(args, 'warmup_sh_degree_trigger'):
            args.warmup_sh_degree_trigger = 2
        if not hasattr(args, 'warmup_sh_iter_trigger'):
            args.warmup_sh_iter_trigger = 1000
        args.enable_warmup = True
    # 信息增益视角选取参数传递
    if args.enable_info_gain:
        setattr(args, 'enable_info_gain', True)
    else:
        setattr(args, 'enable_info_gain', False)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.disabled = args.no_gui
    if not network_gui.disabled:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
