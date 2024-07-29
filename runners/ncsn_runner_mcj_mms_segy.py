import numpy as np
import glob
import tqdm
from losses.dsm import anneal_dsm_score_estimation
import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from models.ncsn import NCSN, NCSNdeeper
from models.UNet import UNet
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_interpolation)
from models import get_sigmas
from models.ema import EMAHelper
from datasets.get_patch import *
from scipy import io as sio

__all__ = ['NCSNRunner']


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA'or config.data.dataset == 'marmousi':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)
    elif config.data.dataset == 'marmousi_unet':
        return UNet(config).to(config.device)


class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):
        # dataset, test_dataset = get_dataset(self.args, self.config)
        # dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
        #                         num_workers=self.config.data.num_workers)
        # test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
        #                          num_workers=self.config.data.num_workers, drop_last=True)
        # test_iter = iter(test_loader)

        dataloaders = []
        # from datasets.seis_mat import SeisMatTrainDataset,SeisMatValidationDataset
        # train_set = SeisMatTrainDataset(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
        #                                 patch_size=self.config.data.image_size, pin_memory=True)

        # Make training data set

        from datasets.prepare_data.mat.bia2small_mat import generate_patch_from_mat
        train_im_list = generate_patch_from_mat(self.args.siesmic_dir, self.args.patch_size, stride=(256, 256))

        # making segy data
        # train_field_list = datagenerator(data_dir=self.args.data_dir, patch_size=self.args.patch_size, stride=self.args.stride,
        #                                  train_data_num=self.args.train_data_num,
        #                                  download=self.args.download, datasets=self.args.datasets, aug_times=self.args.aug_times,
        #                                  scales=self.args.scales,
        #                                  verbose=self.args.verbose_, jump=self.args.jump, agc=False)
        from datasets.prepare_data.segy.big2small_segy import generate_patch_from_segy_1by1
        train_field_list = generate_patch_from_segy_1by1(dir=self.args.data_dir,pch_size=self.args.patch_size,
                                                         stride=self.args.stride,jump=self.args.jump,
                                                         agc=self.args.agc,train_data_num=self.args.train_data_num)
        train_im_list = np.concatenate([train_im_list, train_field_list], axis=0).astype(np.float32)

        # train_set = train_im_list.astype(np.float32)
        from datasets.DenoisingDatasets_seismic import SeisNpyTrainDataset
        train_set=SeisNpyTrainDataset(im_list=train_im_list, length=self.args.cropped_data_num, pch_size=self.args.cropped_patch_size)

        dataloader = DataLoader(train_set, batch_size=self.config.training.batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
        dataloaders.append(dataloader)

        #
        # /home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128  D:\\datasets\\seismic\\marmousi\\f35_s256_o128
        test_loaders = []
        from datasets.seis_mat import SeisMatTrainDataset, SeisMatValidationDataset
        test_set = SeisMatValidationDataset(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
                                        patch_size=self.config.data.image_size, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=self.config.training.batch_size, shuffle=False,
                                  num_workers=0, drop_last=True)
        test_loaders.append(test_loader)
        test_iter = iter(test_loader)
        print('dataloader done!')

        # # Estimate maximum sigma that we should be using
        # compute_approximate_sigma_max = True
        # if compute_approximate_sigma_max:
        #     with torch.no_grad():
        #         current_max_dist = 0
        #         # for i, (X, y) in enumerate(dataloader):
        #         for i, data in enumerate(dataloaders[0]):
        #             X = data['H']
        #             X = X.to(self.config.device)
        #             X = data_transform(self.config, X)
        #             X_ = X.view(X.shape[0], -1)
        #             max_dist = torch.cdist(X_, X_).max().item()
        #
        #             if current_max_dist < max_dist:
        #                 current_max_dist = max_dist
        #             # print(current_max_dist)
        #         print('Final, max eucledian distance using Technique 1: {}'.format(current_max_dist))
        # import math
        # current_max_dist = math.ceil(current_max_dist)
        # print('max eucledian distance after ceiling: {}'.format(current_max_dist))
        # self.config.model.sigma_begin=current_max_dist
        #
        # # # number of steps
        # def get_num_steps(goal, D, sigma_min, sigma_max):
        #     x = np.arange(0, 1, 0.000001)
        #     from scipy.stats import norm
        #     expr = (norm.cdf(np.sqrt(2 * D) * (x - 1) + 3 * x) - norm.cdf(np.sqrt(2 * D) * (x - 1) - 3 * x) - goal) ** 2
        #     i = np.argmin(expr)
        #     alpha = x[i]
        #     n_steps = np.log(sigma_min / sigma_max) / np.log(alpha)
        #     print('Final, Calculate number of steps using Technique 2: {}'.format(n_steps))
        #     return n_steps
        #
        # num_steps=get_num_steps(goal=.5, D=1 * self.config.data.image_size ** 2, sigma_min=self.config.model.sigma_end,
        #               sigma_max=self.config.model.sigma_begin)
        # num_steps=int(math.ceil(num_steps / 50.0)) * 50 #math.ceil(num_steps)
        # print('num_classes after ceiling: {}'.format(num_steps))
        # self.config.model.num_classes= num_steps
        #
        logging.info("Calculated model.sigma_begin: {}".format(self.config.model.sigma_begin))
        logging.info("Calculated model.num_classes: {}".format(self.config.model.num_classes))

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(self.args.resume_ckpt_id)))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        for epoch in range(start_epoch, self.config.training.n_epochs):
            # for i, (X, y) in enumerate(dataloader):
            for i, data in enumerate(dataloaders[0]):
                score.train()
                step += 1
                X= data['H']
                X = X.to(self.config.device)
                X = data_transform(self.config, X)

                loss = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                   self.config.training.anneal_power,
                                                   hook)
                tb_logger.add_scalar('loss', loss, global_step=step)
                tb_hook()

                logging.info("step: {}, loss: {}".format(step, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        # test_X, test_y = next(test_iter)
                        test_data = next(test_iter)
                        test_X = test_data['H']
                    except StopIteration:
                        # test_iter = iter(test_loader)
                        # test_X, test_y = next(test_iter)
                        test_iter = iter(test_loader)
                        test_data = next(test_iter)
                        test_X = test_data['H']

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                                                                    self.config.training.anneal_power,
                                                                    hook=test_hook)
                        tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                        test_tb_hook()
                        logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                        del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        ## Different part from NeurIPS 2019.
                        ## Random state will be affected because of sampling during training time.
                        init_samples = torch.rand(36, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)

                        all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 6)
                        # save_image(image_grid,
                        #            os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))

                        import matplotlib.pyplot as plt
                        plt.gcf().set_size_inches(5, 5)  # 设置图像尺寸为 10x6
                        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
                        # plt.colorbar()  # 添加色标
                        plt.axis('off')  # 关闭坐标轴
                        plt.savefig(os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)), dpi=100,bbox_inches='tight')

                        # torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))
                        sio.savemat(os.path.join(self.args.image_folder, 'samples_{}.mat'.format(step)),
                                    {'data': sample.cpu().squeeze().numpy()})

                        del test_score
                        del all_samples

    def sample(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        # dataset, _ = get_dataset(self.args, self.config)
        # dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
        #                         num_workers=4)
        from datasets.seis_mat import SeisMatTrainDataset, SeisMatValidationDataset
        # 'D:\\datasets\\seismic\\marmousi\\f35_s256_o128'
        dataset = SeisMatTrainDataset(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
                                           patch_size=self.config.data.image_size, pin_memory=True)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)

        score.eval()

        if not self.config.sampling.fid:
            if self.config.sampling.inpainting:
                # data_iter = iter(dataloader)
                # refer_images, _ = next(data_iter)
                data_iter = iter(dataloader)
                samples_ = next(data_iter)
                refer_images = samples_['H']
                refer_images = refer_images.to(self.config.device)
                width = int(np.sqrt(self.config.sampling.batch_size))
                init_samples = torch.rand(width, width, self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)
                all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], score,
                                                                  sigmas,
                                                                  self.config.data.image_size,
                                                                  self.config.sampling.n_steps_each,
                                                                  self.config.sampling.step_lr)

                torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
                refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                     *refer_images.shape[
                                                                                                      1:])
                save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

                if not self.config.sampling.final_only:
                    for i, sample in enumerate(tqdm.tqdm(all_samples)):
                        sample = sample.view(self.config.sampling.batch_size, self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'completion_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'completion_{}.pth'.format(self.config.sampling.ckpt_id)))

            elif self.config.sampling.interpolation:
                if self.config.sampling.data_init:
                    # data_iter = iter(dataloader)
                    # samples, _ = next(data_iter)
                    data_iter = iter(dataloader)
                    samples_ = next(data_iter)
                    samples = samples_['H']
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics_interpolation(init_samples, score, sigmas,
                                                                     self.config.sampling.n_interpolations,
                                                                     self.config.sampling.n_steps_each,
                                                                     self.config.sampling.step_lr, verbose=True,
                                                                     final_only=self.config.sampling.final_only)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, nrow=self.config.sampling.n_interpolations)
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}_inter.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}_inter.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, self.config.sampling.n_interpolations)
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}_inter_1.png'.format(self.config.sampling.ckpt_id)))
                    # torch.save(sample, os.path.join(self.args.image_folder,
                    #                                 'samples_{}_inter_1.pth'.format(self.config.sampling.ckpt_id)))
                    sio.savemat(os.path.join(self.args.image_folder, 'samples_{}_inter_1.mat'.format(self.config.sampling.ckpt_id)),
                                {'data': sample.cpu().squeeze().numpy()})
            else:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=True,
                                                       final_only=self.config.sampling.final_only,
                                                       denoise=self.config.sampling.denoise)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    # torch.save(sample, os.path.join(self.args.image_folder,
                    #                                 'samples_{}.pth'.format(self.config.sampling.ckpt_id)))
                    sio.savemat(os.path.join(self.args.image_folder,
                                             'samples_{}.mat'.format(self.config.sampling.ckpt_id)),
                                {'data': sample.cpu().squeeze().numpy()})

        else:
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size
            if self.config.sampling.data_init:
                dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                        num_workers=4)
                data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.sampling.data_init:
                    try:
                        samples, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples = samples + sigmas_th[0] * torch.randn_like(samples)
                else:
                    samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size, device=self.config.device)
                    samples = data_transform(self.config, samples)

                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       denoise=self.config.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = inverse_data_transform(self.config, img)

                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1

    def test(self):
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas = get_sigmas(self.config)

        dataset, test_dataset = get_dataset(self.args, self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=True)

        verbose = False
        for ckpt in tqdm.tqdm(range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, 5000),
                              desc="processing ckpt:"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            step = 0
            mean_loss = 0.
            mean_grad_norm = 0.
            average_grad_scale = 0.
            for x, y in test_dataloader:
                step += 1

                x = x.to(self.config.device)
                x = data_transform(self.config, x)

                with torch.no_grad():
                    test_loss = anneal_dsm_score_estimation(score, x, sigmas, None,
                                                            self.config.training.anneal_power)
                    if verbose:
                        logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))

                    mean_loss += test_loss.item()

            mean_loss /= step
            mean_grad_norm /= step
            average_grad_scale /= step

            logging.info("ckpt: {}, average test loss: {}".format(
                ckpt, mean_loss
            ))

    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for models with ema
        if self.config.fast_fid.ensemble:
            if self.config.model.ema:
                raise RuntimeError("Cannot apply ensembling to models with EMA.")
            self.fast_ensemble_fid()
            return

        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def fast_ensemble_fid(self):
        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle

        num_ensembles = 5
        scores = [NCSN(self.config).to(self.config.device) for _ in range(num_ensembles)]
        scores = [torch.nn.DataParallel(score) for score in scores]

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            begin_ckpt = max(self.config.fast_fid.begin_ckpt, ckpt - (num_ensembles - 1) * 5000)
            index = 0
            for i in range(begin_ckpt, ckpt + 5000, 5000):
                states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{i}.pth'),
                                    map_location=self.config.device)
                scores[index].load_state_dict(states[0])
                scores[index].eval()
                index += 1

            def scorenet(x, labels):
                num_ckpts = (ckpt - begin_ckpt) // 5000 + 1
                return sum([scores[i](x, labels) for i in range(num_ckpts)]) / num_ckpts

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, scorenet, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)