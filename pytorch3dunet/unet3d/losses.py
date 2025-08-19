import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

from pytorch3dunet.unet3d.utils import get_logger
logger = get_logger('Loss')


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

############################################################################
#############################   ADDED LOSSES   #############################
############################################################################
def compute_per_channel_tversky(input, target, alpha=0.3, beta=0.7, epsilon=1e-6, weight=None):
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    p = flatten(input)
    t = flatten(target).float()

    tp = (p * t).sum(-1)
    p_sum = p.sum(-1)
    t_sum = t.sum(-1)
    fp = p_sum - tp
    fn = t_sum - tp

    if weight is not None:
        tp = weight * tp
        fp = weight * fp
        fn = weight * fn

    denom = (tp + alpha * fp + beta * fn).clamp(min=epsilon)
    return (tp + epsilon) / denom
############################################################################
############################################################################
############################################################################


class MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=1.0):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.bce(input, target) + self.alpha * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = nominator / denominator
        return class_weights.detach()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()

    
############################################################################
#############################   ADDED LOSSES   #############################
############################################################################
class TverskyLoss(_AbstractDiceLoss):
    """
    Tversky loss (1 - T), where T = TP / (TP + alpha * FP + beta * FN).
    Works with binary or multi-class (use normalization='sigmoid' or 'softmax').
    """

    def __init__(self, alpha=0.3, beta=0.7, weight=None, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=weight, normalization=normalization)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        return compute_per_channel_tversky(input, target, alpha=self.alpha, beta=self.beta,
                                           epsilon=self.epsilon, weight=self.weight)


class FocalTverskyLoss(_AbstractDiceLoss):
    """
    Focal Tversky loss: (1 - T) ** gamma, averaged across channels.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, weight=None, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=weight, normalization=normalization)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, input, target):
        probs = self.normalization(input)
        t = compute_per_channel_tversky(probs, target, alpha=self.alpha, beta=self.beta,
                                        epsilon=self.epsilon, weight=self.weight)
        return ((1.0 - t).clamp(min=0.0) ** self.gamma).mean()


class BCETversky(nn.Module):
    """
    BCEWithLogits + lambda * Tversky loss.
    - pos_weight (tensor or float) can be provided to handle imbalance in BCE term.
    - normalization affects the Tversky term only (BCE expects logits).
    """
    def __init__(self, alpha=0.3, beta=0.7, lam=1.0, pos_weight=None, ignore_index=None, epsilon=1e-6):
        super().__init__()
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, logits, target):
        tgt = target.float()
        if self.ignore_index is None:
            mask = torch.ones_like(tgt)
        else:
            mask = target.ne(self.ignore_index).to(dtype=tgt.dtype)

        # BCE (masked)
        bce_el = self.bce(logits, tgt)
        valid = mask.sum().clamp(min=1)
        bce = (bce_el * mask).sum() / valid

        # Tversky (single sigmoid, masked)
        probs = torch.sigmoid(logits)
        t_idx = compute_per_channel_tversky(probs * mask, tgt * mask, alpha=self.alpha, beta=self.beta, epsilon=self.epsilon)
        return bce + self.lam * (1.0 - t_idx.mean())


def sparse_loss(true: torch.Tensor, pred: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """Normalize MSE inside/outside mask separately and sum."""
    true = true.float(); pred = pred.float()
    mask = (true > epsilon).to(dtype=pred.dtype)
    diff2 = (true - pred).pow(2)
    denom_in = mask.sum().clamp(min=1.0)
    denom_out = (mask.numel() - denom_in).clamp(min=1.0)
    loss_in = (diff2 * mask).sum() / denom_in
    loss_out = (diff2 * (1.0 - mask)).sum() / denom_out
    return loss_in + loss_out


def _min_pairwise_distances_chunked(A: torch.Tensor, B: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
    """Min distance from each row in A to any row in B, computed in chunks to reduce memory."""
    device = A.device
    Na = A.shape[0]
    if Na == 0:
        return torch.empty(0, device=device, dtype=A.dtype)
    if B.shape[0] == 0:
        return torch.zeros(Na, device=device, dtype=A.dtype)

    mins = torch.empty(Na, device=device, dtype=A.dtype)
    for i in range(0, Na, chunk_size):
        a_chunk = A[i:i + chunk_size]
        d = torch.cdist(a_chunk, B, p=2)
        mins[i:i + a_chunk.size(0)] = d.min(dim=1).values
        del d
    return mins


def _select_threshold_fast(pred: torch.Tensor, epsilon: float, min_entries: int) -> float:
    """
    Fast, loop-free epsilon optimization:
    - If #(pred > epsilon) >= min_entries: keep epsilon.
    - Else, consider only pred > 0 (since original loop never went below 0).
      Pick the (min_entries)-th largest value among (pred > 0) as the new threshold (minus a tiny delta).
      This guarantees at least min_entries items with '>' comparison, without iteration.
    """
    pred_flat = pred.reshape(-1)
    count_now = (pred_flat > epsilon).sum().item()
    if count_now >= min_entries:
        return float(epsilon)

    pos = pred_flat[pred_flat > 0]
    if pos.numel() < min_entries:
        # Cannot reach min_entries without going below 0; match original behavior by clamping to 0
        return 0.0

    # kth largest among positives → use kthvalue on ascending by selecting index (n-k)
    n = pos.numel()
    k_index = n - min_entries  # 0-based index for kth smallest that corresponds to kth largest overall
    kth = pos.kthvalue(k_index + 1).values.item()
    # Ensure strict '>' keeps at least min_entries (nudge just below kth)
    return max(0.0, min(float(epsilon), float(kth)) - 1e-12)


def _grid_index_positions(spatial_shape, device, dtype):
    """
    Build [N,3] float tensor of voxel indices (z,y,x) for a 3D grid with shape (D,H,W).
    """
    D, H, W = spatial_shape
    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([zz, yy, xx], dim=-1)  # [D,H,W,3]
    return coords.reshape(-1, 3)  # [N,3]


def _distances_single(true3d: torch.Tensor,
                      pred3d: torch.Tensor,
                      epsilon: float,
                      optimize: bool,
                      min_entries: int,
                      chunk_size: int) -> torch.Tensor:
    """
    Compute distances for a single (D,H,W) volume.
    """
    device = pred3d.device
    dtype = pred3d.dtype
    spatial = true3d.shape[-3:]
    grid_pos = _grid_index_positions(spatial, device, dtype)  # [N,3] of indices

    true_flat = true3d.reshape(-1).float()
    pred_flat = pred3d.reshape(-1).float()

    thresh = float(epsilon)
    if optimize:
        thresh = _select_threshold_fast(pred_flat, epsilon=epsilon, min_entries=int(min_entries))

    pred_idx = (pred_flat > thresh).nonzero(as_tuple=False).squeeze(1)
    true_idx = (true_flat > 0).nonzero(as_tuple=False).squeeze(1)

    if pred_idx.numel() == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if true_idx.numel() == 0:
        return torch.zeros(pred_idx.numel(), device=device, dtype=dtype)

    pred_pos = grid_pos.index_select(0, pred_idx)  # [Np, 3]
    true_pos = grid_pos.index_select(0, true_idx)  # [Nt, 3]

    return _min_pairwise_distances_chunked(pred_pos, true_pos, chunk_size=chunk_size)


def distances_from_reco_to_true(
    true: torch.Tensor,
    pred: torch.Tensor,
    epsilon: float = 8.0,
    optimize: bool = False,
    min_entries: int = 20,
    delta_eps: float = 0.1,      # unused (kept for API compatibility)
    chunk_size: int = 65536
) -> torch.Tensor:
    """
    Returns distances from each predicted location (above threshold) to its nearest non-zero true location.
    Works over arbitrary leading dims; the last 3 dims are treated as spatial (D,H,W).
    """
    assert true.shape[-3:] == pred.shape[-3:], "true/pred must share the same spatial shape"
    # Merge any leading dims into a batch B
    B = int(torch.tensor(true.shape[:-3]).numel()) and int(torch.tensor(true.shape[:-3]).prod().item()) or 1
    true_b = true.reshape(B, *true.shape[-3:])
    pred_b = pred.reshape(B, *pred.shape[-3:])

    outs = []
    for b in range(B):
        d = _distances_single(true_b[b], pred_b[b], epsilon=epsilon, optimize=optimize,
                              min_entries=min_entries, chunk_size=chunk_size)
        outs.append(d)

    if len(outs) == 1:
        return outs[0]
    return torch.cat(outs, dim=0)


def distances_from_true_to_reco(true, pred, **kw):
    # Symmetric: swap arguments
    return distances_from_reco_to_true(pred, true, **kw)


def mean_distance_from_reco_to_true(true, pred, **kw):
    d = distances_from_reco_to_true(true, pred, **kw)
    return d.mean() if d.numel() > 0 else d.new_tensor(0.0)


def mean_distance_from_true_to_reco(true, pred, **kw):
    d = distances_from_true_to_reco(true, pred, **kw)
    return d.mean() if d.numel() > 0 else d.new_tensor(0.0)


class SparseLoss(nn.Module):
    def __init__(self, epsilon: float = 1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        return sparse_loss(target, input, epsilon=self.epsilon)


class MeanDistanceFromRecoToTrue(nn.Module):
    """
    Uses integer voxel indices (z,y,x) derived from the tensor's last 3 dims; no grid_pos needed.
    """
    def __init__(self, epsilon: float = 8.0, optimize: bool = False, minEntries: int = 20,
                 chunk_size: int = 65536):
        super().__init__()
        self.epsilon = epsilon
        self.optimize = optimize
        self.minEntries = minEntries
        self.chunk_size = chunk_size

    def forward(self, input, target):
        return mean_distance_from_reco_to_true(
            true=target, pred=input,
            epsilon=self.epsilon, optimize=self.optimize, min_entries=self.minEntries,
            chunk_size=self.chunk_size
        )


class MeanDistanceFromTrueToReco(nn.Module):
    """
    Uses integer voxel indices (z,y,x) derived from the tensor's last 3 dims; no grid_pos needed.
    """
    def __init__(self, epsilon: float = 8.0, optimize: bool = False, minEntries: int = 20,
                 chunk_size: int = 65536):
        super().__init__()
        self.epsilon = epsilon
        self.optimize = optimize
        self.minEntries = minEntries
        self.chunk_size = chunk_size

    def forward(self, input, target):
        return mean_distance_from_true_to_reco(
            true=target, pred=input,
            epsilon=self.epsilon, optimize=self.optimize, min_entries=self.minEntries,
            chunk_size=self.chunk_size
        )


class TotalLoss(nn.Module):
    """Sparse + α·mean(Reco→True) + β·mean(True→Reco) with chunked exact distances and fast epsilon selection.
       Uses voxel index coordinates; no external grid required.
    """
    def __init__(self, epsilon: float = 15.0, optimize: bool = True, minEntries: int = 15,
                 alpha: float = 0.10, beta: float = 0.10, epsilon_sparse: float = 1.0, 
                 chunk_size: int = 65536):
        super().__init__()
        self.sparse = SparseLoss(epsilon=epsilon_sparse)
        self.r2t = MeanDistanceFromRecoToTrue(epsilon=epsilon, optimize=optimize,
                                              minEntries=minEntries, chunk_size=chunk_size)
        self.t2r = MeanDistanceFromTrueToReco(epsilon=epsilon, optimize=optimize,
                                              minEntries=minEntries, chunk_size=chunk_size)
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        return self.sparse(input, target) + self.alpha * self.r2t(input, target) + self.beta * self.t2r(input, target)
############################################################################
############################################################################
############################################################################

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')
    logger.info(f"Creating loss function: {name}")

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        weight = torch.tensor(weight).float()
        logger.info(f"Using class weights: {weight}")

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss


#######################################################################################################################

def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alpha', 1.)
        return BCEDiceLoss(alpha)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return GeneralizedDiceLoss(normalization=normalization)
    elif name == 'DiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        return DiceLoss(weight=weight, normalization=normalization)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'],
                                    initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))
############################################################################
#############################   ADDED LOSSES   #############################
############################################################################
    elif name == 'TverskyLoss':
        alpha = loss_config.get('alpha', 0.3)
        beta = loss_config.get('beta', 0.7)
        normalization = loss_config.get('normalization', 'sigmoid')
        return TverskyLoss(alpha=alpha, beta=beta, weight=weight, normalization=normalization)
    elif name == 'FocalTverskyLoss':
        alpha = loss_config.get('alpha', 0.3)
        beta = loss_config.get('beta', 0.7)
        gamma = loss_config.get('gamma', 0.75)
        normalization = loss_config.get('normalization', 'sigmoid')
        return FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, weight=weight, normalization=normalization)
    elif name == 'BCETversky':
        alpha = loss_config.get('alpha', 0.3)
        beta = loss_config.get('beta', 0.7)
        lam = loss_config.get('lam', 1.0)
        # normalization not needed here; BCE uses logits, Tversky term is internal
        return BCETversky(alpha=alpha, beta=beta, lam=lam, pos_weight=pos_weight)
    elif name == 'SparseLoss':
        epsilon = loss_config.get('epsilon', 1.0)
        return SparseLoss(epsilon=epsilon)
    elif name == 'MeanDistanceFromRecoToTrue':
        epsilon = loss_config.get('epsilon', loss_config.get('eps', 8.0))
        optimize = loss_config.get('optimize', False)
        minEntries = loss_config.get('minEntries', 20)
        chunk_size = loss_config.get('chunk_size', 65536)
        return MeanDistanceFromRecoToTrue(epsilon=epsilon, optimize=optimize,
                                          minEntries=minEntries, chunk_size=chunk_size)
    elif name == 'MeanDistanceFromTrueToReco':
        epsilon = loss_config.get('epsilon', loss_config.get('eps', 8.0))
        optimize = loss_config.get('optimize', False)
        minEntries = loss_config.get('minEntries', 20)
        chunk_size = loss_config.get('chunk_size', 65536)
        return MeanDistanceFromTrueToReco(epsilon=epsilon, optimize=optimize,
                                          minEntries=minEntries, chunk_size=chunk_size)
    elif name == 'TotalLoss':
        epsilon = loss_config.get('epsilon', loss_config.get('eps', 15.0))
        optimize = loss_config.get('optimize', True)
        minEntries = loss_config.get('minEntries', 15)
        alpha = loss_config.get('alpha', 0.10)
        beta = loss_config.get('beta', 0.10)
        epsilon_sparse = loss_config.get('epsilon_sparse', 1.0)
        chunk_size = loss_config.get('chunk_size', 65536)
        return TotalLoss(epsilon=epsilon, optimize=optimize, minEntries=minEntries,
                         alpha=alpha, beta=beta, epsilon_sparse=epsilon_sparse,
                         chunk_size=chunk_size)
############################################################################
############################################################################
############################################################################
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")