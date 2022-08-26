import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import _reduction as _Reduction
from typing import Optional


def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]


class GPLoss(_WeightedLoss):
    
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 CElambda: Optional[Tensor] = None, DLbeta: Optional[Tensor] = None,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0,
                 multiclass: bool = True, num_classes: int = 2) -> None:

        super(GPLoss, self).__init__(weight, size_average, reduce, reduction)
        self.multiclass = multiclass
        self.num_classes = num_classes
        self.weight = weight
        self.CElambda = CElambda
        self.DLbeta = DLbeta

    def update_weight(self, weight: Optional[Tensor] = None):
        self.weight = weight

    def update_lambda(self, CElambda: Optional[Tensor] = None):
        ''' 0 <= lambda <= 1 / (N_ground truth) Union (N_pred)
        '''
        self.CElambda = CElambda

    def update_beta(self, DLbeta: Optional[Tensor] = None):
        ''' 
            P: area of pred
            T: area of ground truth
            DLbeta = P / (T + epsilon)
        '''
        self.DLbeta = DLbeta

    def forward(self, s_input: Tensor, target: Tensor) -> Tensor:

        ce = F.cross_entropy(s_input, target, weight=self.weight)

        input = F.softmax(s_input, dim=1).float()
        target = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        fn = multiclass_dice_coeff if self.multiclass else dice_coeff
        dl = 1 - fn(input, target, reduce_batch_first=True)

        #return ce**self.CElambda + self.DLbeta * dl
        return self.DLbeta * dl

if __name__ == "__main__":

    import torch
    import torch.nn as nn

    loss = GPLoss(weight=torch.tensor([0.02, 0.98]), CElambda=0.5, DLbeta=10.0)
    m = nn.Softmax()

    s_input = torch.randn((5, 2, 256, 256), requires_grad=True)
    t_input = torch.rand((5, 2, 256, 256), requires_grad=True)
    target = torch.randint(0, 2, (5, 256, 256))

    print("s_input: \t{}".format(s_input.size()))
    print("t_input: \t{}".format(t_input.size()))
    print("target: \t{}".format(target.size()))
    print(loss(s_input, target))
