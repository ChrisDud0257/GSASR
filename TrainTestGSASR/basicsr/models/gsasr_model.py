import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import torch.nn.functional as F
import os

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.utils.gaussian_splatting import generate_2D_gaussian_splatting_step
import math
from basicsr.utils.split_and_joint_image import split_and_joint_image


@MODEL_REGISTRY.register()
class GSASRModel(BaseModel):
    def __init__(self, opt):
        super(GSASRModel, self).__init__(opt)
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.tile_size = self.opt['tile_size']

        self.default_step_size = self.opt.get('default_step_size', 1.2)
        self.cuda_rendering = self.opt.get('cuda_rendering', True)
        self.mode = self.opt.get('mode', 'scale_modify')

        self.if_dmax = self.opt.get('if_dmax', False)
        self.dmax_mode = self.opt.get('dmax_mode', 'fix')
        self.dmax = self.opt.get('dmax', 0.1)

        self.denominator = self.opt.get('denominator', 1)


        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            # self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
            net_g = torch.load(load_path)[param_key]
            if self.opt['world_size'] > 1:
                self.net_g.module.load_state_dict(net_g, strict=self.opt['path'].get('strict_load_g', 'False'))
            else:
                self.net_g.load_state_dict(net_g, strict=self.opt['path'].get('strict_load_g', 'False'))

        # define network feature to gaussian splatting
        self.net_fea2gs = build_network(opt['network_fea2gs'])
        self.net_fea2gs = self.model_to_device(self.net_fea2gs)
        self.print_network(self.net_fea2gs)

        # load pretrained models
        load_path = self.opt['path_fea2gs'].get('pretrain_network_fea2gs', None)
        if load_path is not None:
            param_key = self.opt['path_fea2gs'].get('param_key_fea2gs', 'params')
            # self.load_network(self.net_fea2gs, load_path, self.opt['path_fea2gs'].get('strict_load_fea2gs', True), param_key)
            net_fea2gs = torch.load(load_path)[param_key]
            if self.opt['world_size'] > 1:
                self.net_fea2gs.module.load_state_dict(net_fea2gs, strict=self.opt['path_fea2gs'].get('strict_load_fea2gs', 'False'))
            else:
                self.net_fea2gs.load_state_dict(net_fea2gs, strict=self.opt['path_fea2gs'].get('strict_load_fea2gs', 'False'))

        if self.is_train:
            self.clip_grad_norm = self.opt['train'].get('clip_grad_norm', False)
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_fea2gs.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

            self.net_fea2gs_ema = build_network(self.opt['network_fea2gs']).to(self.device)
            # load pretrained model
            load_path = self.opt['path_fea2gs'].get('pretrain_network_fea2gs', None)
            if load_path is not None:
                self.load_network(self.net_fea2gs_ema, load_path, self.opt['path_fea2gs'].get('strict_load_fea2gs', True), 'params_ema')
            else:
                self.model_fea2gs_ema(0)  # copy net_g weight
            self.net_fea2gs_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def model_fea2gs_ema(self, decay=0.999):
        net_fea2gs = self.get_bare_model(self.net_fea2gs)

        net_fea2gs_params = dict(net_fea2gs.named_parameters())
        net_fea2gs_ema_params = dict(self.net_fea2gs_ema.named_parameters())

        for k in net_fea2gs_ema_params.keys():
            net_fea2gs_ema_params[k].data.mul_(decay).add_(net_fea2gs_params[k].data, alpha=1 - decay)

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_type = train_opt['optim_fea2gs'].pop('type')
        self.optimizer_fea2gs = self.get_optimizer(optim_type, self.net_fea2gs.parameters(), **train_opt['optim_fea2gs'])
        self.optimizers.append(self.optimizer_fea2gs)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.scale_train = data['scale'].to(self.device)
        self.gt_size = data['gt_size'].to(self.device)
        # print(f'scale_train is {self.scale_train}')
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'sample_coords' in data:
            self.sample_coords = data['sample_coords'].to(self.device)
        else:
            self.sample_coords = None
        if 'pad_h' in data:
            self.pad_h = data['pad_h'].to(self.device)
        if 'pad_w' in data:
            self.pad_w = data['pad_w'].to(self.device)
        if 'scale_modify' in data:
            self.scale_modify = data['scale_modify']
            

    def feed_val_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt_size = data['gt_size'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.scale = data['scale']
        self.sample_coords = None
        if 'scale_modify' in data:
            self.scale_modify = data['scale_modify']


    def optimize_parameters(self, current_iter):
        b,c = self.gt.shape[:2]
        self.net_g_output = self.net_g(self.lq) #b,c,h,w
        
        scale_vector = []
        for i in range(b):
            scale_vector.append(self.scale_modify[i][0])
        
        scale_vector = torch.tensor(scale_vector).to(self.device)
        batch_gs_parameters = self.net_fea2gs(self.net_g_output, scale_vector)
        # self.output = []

        l_total = 0
        l_pix = 0
        l_ssim = 0
        loss_dict = OrderedDict()
        for i in range(b):
            # b_gt = self.gt[i, :] #c*h_gt*w_gt
            b_lq = self.lq[i, :]
            scale_train = self.scale_train[i]
            gs_parameters = batch_gs_parameters[i,:]
            if self.sample_coords is not None:
                b_sample_coords = self.sample_coords[i]
            else:
                b_sample_coords = None

            
            b_output = generate_2D_gaussian_splatting_step(sr_size=self.gt_size[i], gs_parameters=gs_parameters,
                                                      scale=scale_train,
                                                      sample_coords=b_sample_coords,
                                                      scale_modify = self.scale_modify[i],
                                                      default_step_size = self.default_step_size,
                                                      cuda_rendering=self.cuda_rendering,
                                                      mode = self.mode,
                                                      if_dmax = self.if_dmax,
                                                      dmax_mode = self.dmax_mode,
                                                      dmax = self.dmax).unsqueeze(0)

            if b_sample_coords is None:
                b_output = F.pad(b_output, (0, self.pad_h[i], 0, self.pad_w[i]), 'constant', 0)


            b_gt = self.gt[i].unsqueeze(0)

            # print(f"Original b_gt size is {b_gt.shape}")

            if self.sample_coords is None:
                #since we pad gt to the maximum size, we should remove the padding value here
                b_gt = b_gt[:, :, :self.gt_size[i][0], :self.gt_size[i][1]]
                # print(f"True b_gt size is {b_gt.shape}")
                b_output = b_output[:, :, :self.gt_size[i][0], :self.gt_size[i][1]]

            if self.cri_pix:
                l_pix_b = self.cri_pix(b_output, b_gt)
                l_pix = l_pix + l_pix_b

            if self.cri_ssim:
                l_ssim_b = self.cri_ssim(b_output, b_gt)
                l_ssim = l_ssim + l_ssim_b

        l_pix = l_pix / b
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        if self.cri_ssim:
            l_ssim = l_ssim / b
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim


        l_total = l_total / self.opt['train'].get('accumulation_steps', 1)
        l_total.backward()

        if self.clip_grad_norm:
            # print(f"Clip grad norm")
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(self.net_fea2gs.parameters(), 5)

        # if torch.isnan(l_total).sum() > 0:
        #     print('loss is nan')
        # for i in self.net_g.parameters():
        #     if torch.isnan(i.grad).sum() > 0:
        #         print('net_g grad is nan')
        #     if torch.isnan(i).sum() > 0:
        #         print('net_g value is nan')
        
        # for i in self.net_fea2gs.parameters():
        #     if i.grad is None:
        #         continue
        #     if torch.isnan(i.grad).sum() > 0:
        #         print('net_fea2gs grad is nan')
        #     if torch.isnan(i).sum() > 0:
        #         print('net_fea2gs value is nan')

        if (current_iter + 1) % self.opt['train'].get('accumulation_steps', 1) == 0:
            self.optimizer_g.step()
            self.optimizer_fea2gs.step()

            self.optimizer_g.zero_grad()
            self.optimizer_fea2gs.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            self.model_fea2gs_ema(decay=self.ema_decay)

    def test(self):
        self.lq_pad = self.preprocess(self.lq, self.denominator)
        self.gt_size_pad = torch.tensor([math.ceil(self.scale_modify[0][0] * self.lq_pad.shape[2]), math.ceil(self.scale_modify[0][0] * self.lq_pad.shape[3])])
        self.gt_size_pad = self.gt_size_pad.unsqueeze(0)
        if hasattr(self, 'net_g_ema') and hasattr(self, 'net_fea2gs_ema'):
            self.net_g_ema.eval()
            self.net_fea2gs_ema.eval()
            with torch.no_grad():
                if self.opt['tile_process'] and min(self.lq_pad.shape[2], self.lq_pad.shape[3]) > self.tile_size:
                    self.output = split_and_joint_image(lq = self.lq_pad, scale_factor=self.opt['scale'],
                                                        split_size=self.tile_size,
                                                        overlap_size=self.opt['tile_overlap'],
                                                        model_g=self.net_g_ema,
                                                        model_fea2gs=self.net_fea2gs_ema,
                                                        crop_size=self.opt['crop_size'],
                                                        scale_modify = self.scale_modify[0],
                                                        default_step_size = self.default_step_size,
                                                        cuda_rendering=self.cuda_rendering,
                                                        mode = self.mode,
                                                        if_dmax = self.if_dmax,
                                                        dmax_mode = self.dmax_mode,
                                                        dmax = self.dmax)
                else:
                    self.net_g_output = self.net_g_ema(self.lq_pad)  # b,c,h,w
                    scale_vector = self.scale_modify[0][0].unsqueeze(0).to(self.net_g_output.device)

                    batch_gs_parameters = self.net_fea2gs_ema(self.net_g_output, scale_vector)
                    gs_parameters = batch_gs_parameters[0, :]
                    b_output = generate_2D_gaussian_splatting_step(gs_parameters=gs_parameters,
                                                              sr_size=self.gt_size_pad[0],
                                                              scale = self.opt['scale'],
                                                              sample_coords=None,
                                                              scale_modify = self.scale_modify[0],
                                                              default_step_size = self.default_step_size,
                                                              cuda_rendering=self.cuda_rendering,
                                                              mode = self.mode,
                                                              if_dmax = self.if_dmax,
                                                              dmax_mode = self.dmax_mode,
                                                              dmax = self.dmax)
                    self.output = b_output.unsqueeze(0)

        else:
            self.net_g.eval()
            self.net_fea2gs.eval()
            with torch.no_grad():
                if self.opt['tile_process'] and min(self.lq_pad.shape[2], self.lq_pad.shape[3]) > self.tile_size:
                    self.output = split_and_joint_image(lq=self.lq_pad, scale_factor=self.opt['scale'],
                                                        split_size=self.tile_size,
                                                        overlap_size=self.opt['tile_overlap'],
                                                        model_g=self.net_g,
                                                        model_fea2gs=self.net_fea2gs,
                                                        crop_size=self.opt['crop_size'],
                                                        scale_modify = self.scale_modify[0],
                                                        default_step_size = self.default_step_size,
                                                        cuda_rendering=self.cuda_rendering,
                                                        mode = self.mode,
                                                        if_dmax = self.if_dmax,
                                                        dmax_mode = self.dmax_mode,
                                                        dmax = self.dmax)
                else:
                    self.net_g_output = self.net_g(self.lq_pad)
                    scale_vector = self.scale_modify[0][0].unsqueeze(0).to(self.net_g_output.device)
                    batch_gs_parameters = self.net_fea2gs(self.net_g_output, scale_vector)
                    gs_parameters = batch_gs_parameters[0, :]
                    b_output = generate_2D_gaussian_splatting_step(sr_size = self.gt_size_pad[0],
                                                              gs_parameters=gs_parameters,
                                                              scale = self.opt['scale'],
                                                              sample_coords=None,
                                                              scale_modify = self.scale_modify[0],
                                                              default_step_size = self.default_step_size,
                                                              cuda_rendering=self.cuda_rendering,
                                                              mode=self.mode,
                                                              if_dmax = self.if_dmax,
                                                              dmax_mode = self.dmax_mode,
                                                              dmax = self.dmax)
                    self.output = b_output.unsqueeze(0)
        self.net_g.train()
        self.net_fea2gs.train()
        self.output = self.postprocess(self.output, self.gt_size[0][0], self.gt_size[0][1])

    
    def preprocess(self, x, denominator):
        # pad input image to be a multiple of denominator
        _,c,h,w = x.shape
        if h % denominator > 0:
            pad_h = denominator - h % denominator
        else:
            pad_h = 0
        if w % denominator > 0:
            pad_w = denominator - w % denominator
        else:
            pad_w = 0
        x_new = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        return x_new

    def postprocess(self, x, gt_size_h, gt_size_w):
        x_new = x[:, :, :gt_size_h, :gt_size_w]
        return x_new

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_val_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)


            if with_metrics:
                # calculate metrics
                os.environ['http_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
                os.environ['https_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                del os.environ['http_proxy']
                del os.environ['https_proxy']

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f' \tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema') and hasattr(self, 'net_fea2gs_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
            self.save_network([self.net_fea2gs, self.net_fea2gs_ema], 'net_fea2gs', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_fea2gs, 'net_fea2gs', current_iter)
        self.save_training_state(epoch, current_iter)
