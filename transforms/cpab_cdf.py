import torch
from torch import nn
from .base import Transform
from ..tools import sum_except_batch,_share_across_batch
from splines import unconstrained_cpab_spline, cpab_spline

class ContinuousPiecewiseAffineCDF(Transform):
    def __init__(
            self,
            shape,
            T,
            num_bins=10,
            tails=None,
            tail_bound=1.0,
            identity_init=False,
    ):
        super().__init__()

        self.tail_bound = tail_bound
        self.tails = tails
        self.T = T

        if isinstance(shape, int):
            shape = (shape,)
        # if identity_init: # get unnormalized_widths, bias, and derivatives
        # self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
        self.unnormalized_theta = nn.Parameter(torch.randn(*shape, num_bins-1))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]
        # unnormalized_widths = _share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_theta = _share_across_batch(self.unnormalized_theta, batch_size)
        unnormalized_theta = torch.nan_to_num(unnormalized_theta)
        if self.tails is None:
            spline_fn = cpab_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_cpab_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            # unnormalized_widths = unnormalized_widths,
            unnormalized_theta= unnormalized_theta,
            T=self.T,
            inverse=inverse,
            **spline_kwargs
        )

        return outputs, sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)