
import torch

from datasets.seis_mat import SeisMatTrainDataset, SeisMatValidationDataset
from torch.utils.data import DataLoader
# 'D:\\datasets\\seismic\\marmousi\\f35_s256_o128'
dataset = SeisMatValidationDataset(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
                                   patch_size=128, pin_memory=True)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True,
                        num_workers=0, drop_last=True)
# data_iter = iter(dataloader)
# samples_ = next(data_iter)
# data = samples_['H']

compute_approximate_sigma_max=True
# Estimate maximum sigma that we should be using
if compute_approximate_sigma_max:
    with torch.no_grad():
        current_max_dist = 0
        for i, (X, y) in enumerate(dataloader):
            # X = X.to(self.args.device)
            # X = data_transform(self.config.data, X)
            X_ = X.view(X.shape[0], -1)
            max_dist = torch.cdist(X_, X_).max().item()

            if current_max_dist < max_dist:
                current_max_dist = max_dist
            print(current_max_dist)
        print('Final, max eucledian distance: {}'.format(current_max_dist))
