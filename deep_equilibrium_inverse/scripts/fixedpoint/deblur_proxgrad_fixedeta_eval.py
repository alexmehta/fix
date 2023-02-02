# an evaluation script for a model trained using
# deep_equilibrium_inverse/scripts/fixedpoint/deblur_proxgrad_fixedeta_pre.py
import torch
import os
import random
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import deep_equilibrium_inverse.operators.blurs as blurs
from deep_equilibrium_inverse.operators.operator import OperatorPlusNoise
from deep_equilibrium_inverse.utils.celeba_dataloader import CelebaTrainingDatasetSubset, CelebaTestDataset
from deep_equilibrium_inverse.networks.normalized_equilibrium_u_net import UnetModel
from deep_equilibrium_inverse.solvers.equilibrium_solvers import EquilibriumProxGrad
from deep_equilibrium_inverse.solvers import new_equilibrium_utils as eq_utils
from deep_equilibrium_inverse.utils import cg_utils

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=80)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--and_maxiters', default=100)
parser.add_argument('--and_beta', type=float, default=1.0)
parser.add_argument('--and_m', type=int, default=5)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--data_path', default="/share/data/vision-greg2/mixpatch/img_align_celeba/")
parser.add_argument('--savepath',
                    default="/share/data/vision-greg2/users/gilton/celeba_equilibriumgrad_blur_save_inf.ckpt")
parser.add_argument('--results_file', default="results.csv")
args = parser.parse_args()


# Parameters to modify
n_epochs = int(args.n_epochs)
current_epoch = 0
batch_size = int(args.batch_size)
n_channels = 3
max_iters = int(args.and_maxiters)
anderson_m = int(args.and_m)
anderson_beta = float(args.and_beta)

print_every_n_steps = 2
save_every_n_epochs = 1
initial_eta = 0.2

initial_data_points = 10000
# point this towards your celeba files
data_location = args.data_path

kernel_size = 5
kernel_sigma = 5.0
noise_sigma = 1e-2

# modify this for your machine
save_location = args.savepath

gpu_ids = []
for ii in range(6):
    try:
        torch.cuda.get_device_properties(ii)
        print(str(ii), flush=True)
        if not gpu_ids:
            gpu_ids = [ii]
        else:
            gpu_ids.append(ii)
    except (AssertionError, RuntimeError):
        print('Not ' + str(ii) + "!", flush=True)

print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
gpu_ids = [int(x) for x in gpu_ids]
# device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

# Set up data and dataloaders
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
celeba_train_size = 162770
total_data = initial_data_points
if args.debug:
    # take only a few data points for debugging
    total_indices = random.sample(range(celeba_train_size), k=3*batch_size)
    initial_indices = total_indices
    try:
        import lovely_tensors as lt
    except ImportError:
        pass
    else:
        lt.monkey_patch()
else:
    total_indices = random.sample(range(celeba_train_size), k=total_data)
    initial_indices = total_indices

dataset = CelebaTrainingDatasetSubset(data_location, subset_indices=initial_indices, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True,
)

test_dataset = CelebaTestDataset(data_location, transform=transform)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
)
print("loaded data")

### Set up solver and problem setting

forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                      n_channels=3, n_spatial_dimensions=2).to(device=device)
measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

internal_forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                      n_channels=3, n_spatial_dimensions=2).to(device=device)

# standard u-net
learned_component = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32)

solver = EquilibriumProxGrad(linear_operator=internal_forward_operator, nonlinear_operator=learned_component,
                    eta=initial_eta, minval=-1, maxval = 1)

if use_dataparallel:
    solver = nn.DataParallel(solver, device_ids=gpu_ids)
solver = solver.to(device=device)

cpu_only = not torch.cuda.is_available()


if os.path.exists(save_location):
    if not cpu_only:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')

    start_epoch = saved_dict['epoch']
    try:
        solver.load_state_dict(saved_dict['solver_state_dict'])
    except RuntimeError:
        # we are using data parallel and the saved model is not
        solver.module.load_state_dict(saved_dict['solver_state_dict'])


# set up loss and train
lossfunction = torch.nn.MSELoss(reduction='sum')

forward_iterator = eq_utils.andersonexp
deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, m=anderson_m, beta=anderson_beta, lam=1e-2,
                                        max_iter=max_iters, tol=1e-5)

# Do train
print("starting eval")
losses = []
initial_losses = []
# use a pandas dataframe to save the results
# the dataframe will be saved to a csv file at the end of evaluation
# the csv file should be created if it does not exist
# otherwise we should just append to it
# the columns should contain the maximum number of anderson steps
# the loss and the initial loss
results_file = args.results_file
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
else:
    results_df = pd.DataFrame(columns=['max_iters', 'loss', 'initial_loss'])
for ii, sample_batch in enumerate(test_dataloader):
    sample_batch = sample_batch.to(device=device)
    y = measurement_process(sample_batch)
    if forward_operator is not None:
        with torch.no_grad():
            initial_point = cg_utils.conjugate_gradient(initial_point=forward_operator.adjoint(y),
                                                            ATA=forward_operator.gramian,
                                                            regularization_lambda=noise_sigma, n_iterations=60)
            initial_loss = lossfunction(initial_point, sample_batch)
            initial_losses.append(initial_loss.item())
        reconstruction = deep_eq_module.forward(y, initial_point=initial_point)
    else:
        reconstruction = deep_eq_module.forward(y)
    loss = lossfunction(reconstruction, sample_batch)
    losses.append(loss.item())
mean_loss = np.mean(losses)
mean_initial_loss = np.mean(initial_losses)
print("mean loss: " + str(mean_loss))
print("mean initial loss: " + str(mean_initial_loss))
results_df = results_df.append({'max_iters': max_iters, 'loss': mean_loss, 'initial_loss': mean_initial_loss}, ignore_index=True)
results_df.to_csv(results_file, index=False)
