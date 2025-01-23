import torch
import torch.nn as nn
import torch.nn.functional as F

def cat(xs, dim=-1):
    return torch.cat(xs, dim=dim)

def expand_dims(x, dim=-1):
    return x.unsqueeze(dim)

def layer_norm(xs):
    layer_norm = nn.LayerNorm(xs[0].size())
    return [layer_norm(x) for x in xs]

def squeeze(x, dim=None):
    if dim is None:
        return x.squeeze()
    else:
        return x.squeeze(dim)

def sum(x, dim=None, include_batch_dim=False):
    if isinstance(x, list):
        return torch.sum(torch.stack(x), dim=0)
    if dim is None:
        return torch.sum(x)
    else:
        return torch.sum(x, dim=dim, keepdim=include_batch_dim)

def mean(x, dim=None, include_batch_dim=False):
    if isinstance(x, list):
        return torch.mean(torch.stack(x), dim=0)
    if dim is None:
        return torch.mean(x)
    else:
        return torch.mean(x, dim=dim, keepdim=include_batch_dim)

def split(x, dim=1):
    return torch.split(x, 1, dim=dim)

def pick_mat(x, row_idx, col_idx):
    return x[row_idx, col_idx]

def logsumexp_dim(x, dim=0):
    return torch.logsumexp(x, dim=dim)

def log_sum_exp(scores, n_tags):
    max_score = scores.max(dim=1, keepdim=True).values
    return max_score + torch.log(torch.sum(torch.exp(scores - max_score), dim=1, keepdim=True))

def dropout_list(rep_list, dp_rate):
    return [F.dropout(rep, p=dp_rate, training=True) for rep in rep_list]

def dropout_dim_list(rep_list, dp_rate, dim=0):
    return [F.dropout(rep, p=dp_rate, training=True) for rep in rep_list]

def cat_list(rep_list_a, rep_list_b, dim=0):
    return [torch.cat([rep_a, rep_b], dim=dim) for rep_a, rep_b in zip(rep_list_a, rep_list_b)]

def add_list(rep_list_a, rep_list_b):
    return [rep_a + rep_b for rep_a, rep_b in zip(rep_list_a, rep_list_b)]

def sum_list(rep_list_a, rep_list_b):
    return [rep_a + rep_b for rep_a, rep_b in zip(rep_list_a, rep_list_b)]

def binary_cross_entropy(x, y):
    return F.binary_cross_entropy_with_logits(x, y)

def max_np(np_vec):
    np_vec = np_vec.flatten()
    return np_vec.max().item(), np_vec.argmax().item()
