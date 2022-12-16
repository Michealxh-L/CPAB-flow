import torch

def is_int(x):
    return isinstance(x, int)

def is_nonnegative_int(x):
    return is_int(x) and x >= 0

def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def _share_across_batch(params, batch_size):
    return params[None,...].expand(batch_size, *params.shape)