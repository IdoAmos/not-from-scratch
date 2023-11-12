import math
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial

def _student_t_map(mu, sigma, nu):
    sigma = F.softplus(sigma)
    nu = 2.0 + F.softplus(nu)
    return mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)

def student_t_loss(outs, y):
    mu, sigma, nu = outs[..., 0], outs[..., 1], outs[..., 2]
    mu, sigma, nu = _student_t_map(mu, sigma, nu)
    y = y.squeeze(axis=-1)

    nup1_half = (nu + 1.0) / 2.0
    part1 = 1.0 / nu * torch.square((y - mu) / sigma)
    Z = (
        torch.lgamma(nup1_half)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(math.pi * nu)
        - torch.log(sigma)
    )

    ll = Z - nup1_half * torch.log1p(part1)
    return -ll.mean()

def gaussian_ll_loss(outs, y):
    mu, sigma = outs[..., 0], outs[..., 1]
    y = y.squeeze(axis=-1)
    sigma = F.softplus(sigma)
    ll = -1.0 * (
        torch.log(sigma)
        + 0.5 * math.log(2 * math.pi)
        + 0.5 * torch.square((y - mu) / sigma)
    )
    return -ll.mean()

def binary_cross_entropy(logits, y):
    # BCE loss requires squeezing last dimension of logits so it has the same shape as y
    # requires y to be float, since it's overloaded to represent a probability
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())


def binary_accuracy(logits, y):
    return torch.eq(logits.squeeze(-1) >= 0, y).float().mean()


def cross_entropy(logits, y, weighted=False, ignore_index=-100):
    C = logits.shape[-1]
    logits = logits.reshape(-1, C)
    y = y.reshape(-1)
    weight = None 
    if weighted:
        weight = y.new_zeros(C, dtype=logits.dtype)
        classes, counts = y.unique(sorted=True, return_counts=True)
        weight[classes] = 1 / counts
    return F.cross_entropy(logits, y, weight, ignore_index=ignore_index)


def soft_cross_entropy(logits, y, **kwargs):
    logits = logits.view(-1, logits.shape[-1])
    # target is now 2d (no target flattening)
    return F.cross_entropy(logits, y, **kwargs)


def accuracy(logits, y, ignore_index=None, balanced=False):
    logits = logits.reshape(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    preds = torch.argmax(logits, dim=-1)
    
    if balanced:
        assert ignore_index is None
        return balanced_accuracy_score(y.cpu().data, preds.cpu().data)
        
    if ignore_index is None:
        return (preds == y).float().mean()
        
    err = ((preds != y) & (y != ignore_index)).float().sum()
    count = (y != ignore_index).float().sum().clamp_min(1e-4)
    return 1 - err / count


def accuracy_at_k(logits, y, k=1):
    logits = logits.view(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.topk(logits, k, dim=-1)[1].eq(y.unsqueeze(-1)).any(dim=-1).float().mean()


def f1_binary(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="binary")


def f1_macro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="macro")


def f1_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro")


def roc_auc_macro(logits, y):
    logits = logits.view(
        -1, logits.shape[-1]
    ).detach()  # KS: had to add detach to eval while training
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="macro"
    )


def roc_auc_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return roc_auc_score(
        y.cpu().numpy(), F.softmax(logits, dim=-1).cpu().numpy()[:, 1], average="micro"
    )

    
def mse(outs, y, len_batch=None, r2=False):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.mse_loss(outs, y) if not r2 else r2_score(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked) if not r2 else r2_score(outs_masked, y_masked)


def masked_mse(outs, y, len_batch=None, r2=False, ignore_value=-10000.0):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)

    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)

    # remove entries with ignore_value
    mask = y != ignore_value
    outs = outs[mask]
    y = y[mask]

    if len_batch is None:
        return F.mse_loss(outs, y) if not r2 else r2_score(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked) if not r2 else r2_score(outs_masked, y_masked)


def r2_score(outs, y, output_wise=False, eps=1e-5):
    """computes batch-level/output-position-wise r2 score if possible else reverts to batch-level r2"""
    def batch_r2(outs, y):
        return 1 - F.mse_loss(outs, y) / F.mse_loss(y.mean(), y).clamp_min(eps)
    
    if not output_wise or y.ndim == 1:
        return batch_r2(outs.reshape(-1), y.reshape(-1))
    
    outs, y = outs.flatten(0,-2), y.flatten(0,-2)
    
    if y.size(0) == 1:
        return batch_r2(outs.reshape(-1), y.reshape(-1))
        
    output_wise_r2 = 1 - (outs - y).pow(2).mean(0) / (y.mean(0, keepdim=True) - y).pow(2).mean(0).clamp_min(eps)
    return output_wise_r2.mean()


def forecast_rmse(outs, y, len_batch=None):
    # TODO: generalize, currently for Monash dataset
    return torch.sqrt(F.mse_loss(outs, y, reduction='none').mean(1)).mean()

def mae(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)

def masked_mae(outs, y, len_batch=None, ignore_value=-10000.0):
    """
    Computes the mean absolute error of the first `len_batch` items in the batch.
    ignores indices with label=ignore_value
    """
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)

    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)

    # remove entries with ignore_value
    mask = y != ignore_value
    outs = outs[mask]
    y = y[mask]

    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)

def frequency_weights(y):
    """
    generate weights for each label according to inverse ratio of each labels
    """
    # mapping between every unique value in y1 to an integer
    vals, y_int, counts = torch.unique(y, return_inverse=True, return_counts=True)

    # Calculate the occurrences of each value in y_int - occurence of each unique value
    occurrences = torch.bincount(y_int)

    # Create the weight array
    w = torch.reciprocal(occurrences.float())

    # Assign weights to corresponding values in y_int and normalize
    weights = w[y_int]
    weights = weights / torch.sum(weights)
    return weights

def masked_wmae(outs, y, len_batch=None, ignore_value=-10000.0):
    """
    Computes the mean absolute error of the first `len_batch` items in the batch.
    ignores indices with label=ignore_value
    """
    assert len_batch is None, "in masked_wmae len_batch is not supported"

    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)

    # remove entries with ignore_value
    mask = y != ignore_value
    outs = outs[mask]
    y = y[mask]

    # add weights according to inverse ratio of each labels
    w = frequency_weights(y)

    loss = torch.abs(outs - y) * w
    return loss.mean()


# Metrics that can depend on the loss
def loss(x, y, loss_fn):
    """ This metric may be useful because the training loss may add extra regularization (e.g. weight decay implemented as L2 penalty), while adding this as a metric skips the additional losses """
    return loss_fn(x, y)


def bpb(x, y, loss_fn):
    """ bits per byte (image density estimation, speech generation, char LM) """
    return loss_fn(x, y) / math.log(2)


def ppl(x, y, loss_fn):
    return torch.exp(loss_fn(x, y))


# should have a better way to do this
output_metric_fns = {
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "binary_accuracy": binary_accuracy,
    "accuracy": accuracy,
    'accuracy_ignore_m100': partial(accuracy, ignore_index=-100),
    'accuracy@3': partial(accuracy_at_k, k=3),
    'accuracy@5': partial(accuracy_at_k, k=5),
    'accuracy@10': partial(accuracy_at_k, k=10),
    "eval_loss": loss,
    "mse": mse,
    "masked_mse": masked_mse,
    "mae": mae,
    "masked_mae": masked_mae,
    "masked_wmae": masked_wmae,
    "r2": partial(mse, r2=True),
    "masked_r2": partial(masked_mse, r2=True),
    "forecast_rmse": forecast_rmse,
    "f1_binary": f1_binary,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "roc_auc_macro": roc_auc_macro,
    "roc_auc_micro": roc_auc_micro,
    "soft_cross_entropy": soft_cross_entropy,  # only for pytorch 1.10+
    "student_t": student_t_loss,
    "gaussian_ll": gaussian_ll_loss,
}

try:
    from segmentation_models_pytorch.utils.functional import iou
    from segmentation_models_pytorch.losses.focal import focal_loss_with_logits

    def iou_with_logits(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
        return iou(pr.sigmoid(), gt, eps=eps, threshold=threshold, ignore_channels=ignore_channels)

    output_metric_fns["iou"] = partial(iou, threshold=0.5)
    output_metric_fns["iou_with_logits"] = partial(iou_with_logits, threshold=0.5)
    output_metric_fns["focal_loss"] = focal_loss_with_logits
except ImportError:
    pass

loss_metric_fns = {
    "loss": loss,
    "bpb": bpb,
    "ppl": ppl,
}
metric_fns = {**output_metric_fns, **loss_metric_fns}  # TODO py3.9

