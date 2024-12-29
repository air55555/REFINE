import os
from pathlib import Path

import matplotlib.pyplot as plt

import torch
import yaml
import spectral


from dphsir.denoisers import (FFDNet3DDenoiser, FFDNetDenoiser, QRNN3DDenoiser,
                              TVDenoiser, GRUNetDenoiser, IRCNNDenoiser,
                              DRUNetDenoiser, GRUNetTVDenoiser)
from dphsir.solvers import callbacks
from dphsir.solvers.params import admm_log_descent
from dphsir.metrics import mpsnr, mssim, sam, ergas
from dphsir.utils.io import loadmat
from utils import *

def get_denoiser(cfg):
    if cfg.type.startswith('qrnn3d'):
        use_noise_map = 'map' in cfg.type
        return QRNN3DDenoiser(cfg.model_path, use_noise_map=use_noise_map)
    elif cfg.type.startswith('grunettv'):
        return GRUNetTVDenoiser(cfg.model_path)
    elif cfg.type.startswith('grunet'):
        return GRUNetDenoiser(cfg.model_path)
    elif cfg.type.startswith('drunet'):
        return DRUNetDenoiser(1, cfg.model_path)
    elif cfg.type.startswith('ircnn'):
        return IRCNNDenoiser(1, cfg.model_path)
    elif cfg.type.startswith('ffdnet'):
        if '3d' in cfg.type:
            return FFDNet3DDenoiser(cfg.model_path)
        else:
            return FFDNetDenoiser(1, cfg.model_path)
    elif cfg.type.startswith('tv'):
        return TVDenoiser(iter_num=5, use_3dtv='3d' in cfg.type)
    else:
        raise ValueError('Unsupported denoiser, choices \
            [qrnn3d, qrnn3d_map, grunettv, grunet, drunet, ircnn, ffdnet, ffdnet3d, tv]')


def get_params(cfg):
    rhos, sigmas = admm_log_descent(sigma=max(0.255/255., 0),
                                    iter_num=cfg.iter,
                                    modelSigma1=cfg.sigma1, modelSigma2=cfg.sigma2,
                                    w=cfg.w,
                                    lam=cfg.lam)
    return rhos, sigmas


def format(d): return ' '.join(['{}: {:.4f}'.format(k, v) for k, v in d.items()])


def show_results(input, pred, gt, lowres,output_path=None, ):
    input = input.astype('float32')
    pred = pred.astype('float32')
    gt = gt.astype('float32')

    #in case for misr use only part of task.run return
    if isinstance(lowres, tuple): lowres=lowres[0]

    VISUAL_CHANNEL = 9
    def hsi2rgb(x): return x[:, :, VISUAL_CHANNEL]
    CMAP = 'gray'
    if output_path:
        print('-----------------------------------------------------')
        print(f'Saving to ({output_path})')
        output_path = Path(output_path)
        if output_path.suffix:
            img = [hsi2rgb(i) for i in [input, pred, gt]]
            plt.imsave(output_path, np.hstack(img), cmap=CMAP)
        else:
            os.makedirs(output_path, exist_ok=True)
            plt.imsave(output_path/'input.png', hsi2rgb(input), cmap=CMAP)
            plt.imsave(output_path/'lowres.png', hsi2rgb(lowres), cmap=CMAP)
            plt.imsave(output_path/'gt.png', hsi2rgb(gt), cmap=CMAP)
            plt.imsave(output_path/'pred.png', hsi2rgb(pred), cmap=CMAP)
            spectral.envi.save_image(output_path/'input.hdr', input, dtype='float32', force=True)
            spectral.envi.save_image(output_path / 'pred.hdr', pred, dtype='float32', force=True)

    else:
        img = [hsi2rgb(i) for i in [input, pred, gt]]
        plt.imshow(np.hstack(img), cmap=CMAP)
        plt.show()
    metrics = [mpsnr, mssim, sam, ergas]

    def eval(inp, gt):
        return {m.__name__: m(inp, gt) for m in metrics}

    before_eval = eval(input, gt)
    after_eval = eval(pred, gt)
    print('-----------------------------------------------------')
    print('Before |', format(before_eval))
    print(' After |', format(after_eval))
    with open(r'..\log.txt', "a") as file:
        file.write(f'Before | {before_eval}\n')
        file.write(f' After | {after_eval}\n')
    return eval(pred, gt)


def restore(task, cfg):
    print('-----------------------------------------------------')
    yaml.emitter.Emitter.prepare_tag = lambda self, tag: ''
    print(yaml.dump(cfg), end='')
    with open(r'..\log.txt', "a") as file:
        file.write("***************\n")  # Optional: Add a newline between runs
        yaml.dump(cfg, file, default_flow_style=False)

    print('-----------------------------------------------------')
    device = torch.device(cfg.device)

    def run(input_path, output_path):
        data = loadmat(input_path)
        gt = data['gt'].astype(np.float32)
        #one time run to save lehavim as envi
        #save_hsi_as(gt, "../../hsi_cheese/lehavim.hdr")

        if cfg.t == 'no_gt':
            gt_safe = gt
            gt = super_resolve_hsi(gt,cfg.sf)
        input, init, solver = task(gt, device, cfg)

        print('GT:', gt.shape)

        rhos, sigmas = get_params(cfg.params)
        iter = len(rhos)
        #real task - sr without gt  dont know what to find out
        #if cfg.t =='no_gt' : input = gt_safe
        pb = callbacks.ProgressBar(iter)
        pred = solver.restore(input, iter_num=iter, rhos=rhos, sigmas=sigmas,
                              callbacks=[pb])
        #write totals to log etc.
        pb.close()

        #if cfg.t == 'no_gt':
        #    input = restore_hsi(gt,cfg.sf)
        return show_results(init, pred, gt, input, output_path)

    if os.path.isdir(cfg.input_path):
        paths = list(Path(cfg.input_path).glob('*.mat'))
        print('Test directory:', cfg.input_path)
        print('Found {} files'.format(len(paths)))
        total_metrics = {}
        count = 0
        for input_path in paths:
            output_path = Path(cfg.output_path)/input_path.stem
            print('Processing {}'.format(input_path))
            metrics = run(input_path, output_path)
            for k, v in metrics.items():
                total_metrics.setdefault(k, 0)
                total_metrics[k] += v
            count += 1
        print('Total {} files'.format(len(paths)))
        print('Average metrics:', format({k: v/count for k, v in total_metrics.items()}))
    else:
        run(cfg.input_path, cfg.output_path)
