from nflows.nn import nets
from nflows.utils import torchutils
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation

import sklearn.datasets as datasets
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import difw

from ..transforms import CPABCouplingTransform

def create_coupling_transform(cls, shape, **kwargs):
    if len(shape) == 1:

        def create_net(in_features, out_features):
            return nets.ResidualNet(
                in_features, out_features, hidden_features=64, num_blocks=2
            )

    else:
        def create_net(in_channels, out_channels):
            return nets.ConvResidualNet(
                in_channels=in_channels, out_channels=out_channels, hidden_channels=16
            )

    mask = torchutils.create_mid_split_binary_mask(shape[0])

    return cls(mask=mask, transform_net_create_fn=create_net, **kwargs), mask


#%% Create datasets
train_set,_ = datasets.make_moons(1000, noise=.1)
train_set = (train_set - np.min(train_set))/(np.max(train_set) - np.min(train_set))
training_loader = torch.utils.data.DataLoader(torch.Tensor(train_set).cuda(), batch_size=128, shuffle=True)
plt.scatter(train_set[:, 0], train_set[:, 1])
plt.show()

#%% Build CPAB flow
num_layers = 7
base_dist = StandardNormal(shape=[2])
shape = [2]
num_bins = 4

T = difw.Cpab(tess_size= num_bins, backend="pytorch", device="gpu", zero_boundary=True, basis="qr")
params = T.params

transforms = []
for _ in range(num_layers):
    transform, _ = create_coupling_transform(CPABCouplingTransform, shape, num_bins= num_bins, tails = 'linear',tail_bound = 5, apply_unconditional_transform = True)
    transforms.append(transform)
    transforms.append(ReversePermutation(features=2))

transforms = CompositeTransform(transforms)
flow = Flow(transforms, base_dist)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-3)

# Move model on GPU if available
enable_cuda = True
gpu = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
flow = flow.to(gpu)
cpu = torch.device('cpu')

#%% Training
# CPAB difw transform with closed form without shrink
num_iter = 300
loss_hist = []
for i in tqdm(range(num_iter)):
  loss_value = []
  for j, data in enumerate(training_loader):
    # with torch.autograd.detect_anomaly():
      optimizer.zero_grad()
      x = data.clone().detach().float().to(gpu)
      loss = -flow.log_prob(inputs=x).mean()
      loss_value.append(loss.item())
      loss.backward()
      optimizer.step()
  mean_loss = sum(loss_value)/len(loss_value)
  loss_hist.append(mean_loss)
  # print(mean_loss)
  if (i + 1) % 20 == 0:
    print(mean_loss)
    xline = torch.linspace(-1, 2,100)
    yline = torch.linspace(-1, 2,100)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    xyinput = xyinput.to(gpu)
    with torch.no_grad():
        zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)
    zgrid = zgrid.cpu()
    plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
    plt.title('iteration {}'.format(i + 1))
    plt.show()
    # generate samples
    with torch.no_grad():
      generated_x = flow.sample(500).cpu().numpy()
    plt.scatter(generated_x[:, 0], generated_x[:, 1])
    plt.show()
plt.plot(loss_hist)
plt.show()

