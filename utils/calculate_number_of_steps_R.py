## Calculate number of steps using Technique 2 from https://arxiv.org/abs/2006.09011
## sigma_min should always be .01
## Use "main.py --compute_approximate_sigma_max" to get sigma_max
## D is the dimension
## Goal should be .50 or .99

import numpy as np
from scipy.stats import norm

x = np.arange(0, 1, 0.000001)
D = 1.5  # 假设给定的D值
goal = 0.5  # 假设给定的goal值
sigma_min = 0.1  # 假设给定的sigma_min值
sigma_max = 1.0  # 假设给定的sigma_max值
def get_num_steps(goal, D, sigma_min, sigma_max):
	expr = (norm.cdf(np.sqrt(2 * D) * (x - 1) + 3 * x) - norm.cdf(np.sqrt(2 * D) * (x - 1) - 3 * x) - goal) ** 2
	i = np.argmin(expr)
	alpha = x[i]
	n_steps = np.log(sigma_min / sigma_max) / np.log(alpha)
	return n_steps

## CIFAR-10
# num_steps=get_num_steps(.5, 3*32*32, .01, 50) # around 225 steps for C = .50

num_steps=get_num_steps(.5, 1*128*128, .001, 23)
print('num_steps:',num_steps)
# get_num_steps(.99, 3*32*32, .01, 50) # around 1k steps for C = .99
#
# ## LSUN-Churches
# get_num_steps(.5, 3*64*64, .01, 140) # around 500 steps for C = .50
# get_num_steps(.99, 3*64*64, .01, 140) # around 2.2k steps for C = .99
