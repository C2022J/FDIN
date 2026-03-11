import torch
from pytorch_msssim import SSIM
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.models.archs.vgg_arch import VGGFeatureExtractor
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, reduction_override=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
            reduction_override (str, optional): If provided, overrides the instance's reduction mode
                                                for this forward pass. Default: None.
        """
        # <<< 修改点 1: 确定本次 forward 使用的 reduction 模式 >>>
        current_reduction = reduction_override if reduction_override is not None else self.reduction
        if current_reduction not in ['none', 'mean', 'sum']:
             raise ValueError(f'Unsupported reduction override: {reduction_override}.')
        # <<< 修改结束 >>>

        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=current_reduction) # <--- 使用 current_reduction

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class SSIMLoss(nn.Module):
    """SSIM (Structural Similarity) Loss using pytorch-msssim.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
        window_size (int): The size of the Gaussian window. Default: 11.
        size_average (bool): If True, average the loss over the batch. Default: True.
        channel (int): Number of image channels. Default: 3.
    """

    def __init__(self, loss_weight=1.0, window_size=11, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ssim_module = SSIM(
            data_range=1.0,
            size_average=size_average,
            channel=channel,
            win_size=window_size
        )

    def forward(self, pred, target):
        ssim_value = self.ssim_module(pred, target)
        loss = 1 - ssim_value
        return self.loss_weight * loss

class L1SSIMLoss(nn.Module):
    """组合 L1 和 SSIM 的损失函数。

    Args:
        loss_weight (float): 总的损失权重。默认值：1.0。
        l1_weight (float): L1损失的权重。默认值：0.5。
        ssim_weight (float): SSIM损失的权重。默认值：0.5。
        reduction (str): 指定应用于输出的 reduction 方法。选项有 'none'、'mean'、'sum'。默认值：'mean'。
        window_size (int): SSIM的高斯窗口大小。默认值：11。
        channel (int): 输入图像的通道数。默认值：3。
    """

    def __init__(self, loss_weight=1.0, l1_weight=0.5, ssim_weight=0.5, reduction='mean', window_size=11, channel=3):
        super(L1SSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: none, mean, sum.')
        self.loss_weight = loss_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = L1Loss(loss_weight=1.0, reduction=reduction)
        self.ssim_loss = SSIMLoss(loss_weight=1.0, window_size=window_size, channel=channel)

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 预测的张量，形状为 (N, C, H, W)。
            target (Tensor): 目标张量，形状为 (N, C, H, W)。
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        combined_loss = self.l1_weight * l1 + self.ssim_weight * ssim
        return self.loss_weight * combined_loss


class L1L2Loss(nn.Module):
    """组合 L1 和 L2 (MSE) 的损失函数。

    Args:
        loss_weight (float): 总的损失权重。默认值：1.0。
        l1_weight (float): L1损失的权重。默认值：0.5。
        l2_weight (float): L2 (MSE) 损失的权重。默认值：0.5。
        reduction (str): 指定应用于输出的 reduction 方法。选项有 'none'、'mean'、'sum'。默认值：'mean'。
    """

    def __init__(self, loss_weight=1.0, l1_weight=0.5, l2_weight=0.5, reduction='mean'):
        super(L1L2Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: none, mean, sum.')

        self.loss_weight = loss_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

        # 实例化 L1Loss 和 MSELoss
        # 注意：这里的内部 loss_weight 设置为 1.0，因为最终的权重将在 forward 方法中统一应用
        self.l1_loss = L1Loss(loss_weight=1.0, reduction=reduction)
        self.l2_loss = MSELoss(loss_weight=1.0, reduction=reduction)

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 预测的张量，形状为 (N, C, H, W)。
            target (Tensor): 目标张量，形状为 (N, C, H, W)。
        """
        # 分别计算 L1 和 L2 损失
        l1 = self.l1_loss(pred, target)
        l2 = self.l2_loss(pred, target)

        # 根据权重加权求和
        combined_loss = (self.l1_weight * l1) + (self.l2_weight * l2)

        # 应用总的损失权重
        return self.loss_weight * combined_loss

# [ 把以下代码粘贴到你 losses.py 文件的末尾 ]

@weighted_loss
def frequency_charbonnier_loss(pred, target, eps=1e-6):
    """
    计算频率域的沙博尼耶损失 (L_FC)
    这被 RIISSR 和 MFFSSR 使用。
    """
    # 1. 使用 FFT 将图像转换到频率域
    # 我们使用 rfft2 (实数FFT)，因为它对实数图像（如RGB）更高效
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')

    # 2. 计算复数差异的平方幅度 (magnitude squared)
    # (pred_real - target_real)^2 + (pred_imag - target_imag)^2
    diff = pred_fft - target_fft
    magnitude_sq_diff = diff.real ** 2 + diff.imag ** 2

    # 3. 应用 Charbonnier (sqrt(x^2 + eps^2))
    # 注意: 我们已经有了 diff^2, 所以我们只需要计算 sqrt(magnitude_sq_diff + eps^2)
    # (为了与MFFSSR/RIISSR的定义保持一致，eps应该在sqrt内部平方)
    loss = torch.sqrt(magnitude_sq_diff + (eps ** 2))

    # 返回未 reduce 的损失，交给 @weighted_loss 装饰器处理
    return loss


class FrequencyCharbonnierLoss(nn.Module):
    """频率沙博尼耶损失 (L_FC)，用于超分任务。
    它强制网络不仅匹配空间像素，还要匹配高频纹理成分。

    Args:
        loss_weight (float): 损失权重. Default: 1.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): 用于稳定 Charbonnier 损失的小常数. Default: 1e-6.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-6):
        super(FrequencyCharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): (N, C, H, W)
            target (Tensor): (N, C, H, W)
        """
        return self.loss_weight * frequency_charbonnier_loss(
            pred,
            target,
            weight=weight,
            reduction=self.reduction,
            eps=self.eps)

class CrossEntropyLoss(nn.Module):
    """交叉熵损失，用于任务分类辅助损失。

    Args:
        loss_weight (float): 损失权重. Default: 1.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        label_smoothing (float): 标签平滑系数. Default: 0.0.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pred_logits, target_labels, weight=None, **kwargs):
        """
        Args:
            pred_logits (Tensor): 模型预测的原始 logits, shape (N, num_tasks).
            target_labels (Tensor): 真实的整数任务标签, shape (N).
            weight (Tensor, optional): 样本权重, shape (N,). Default: None.
        """
        # 注意：F.cross_entropy 内部包含了 Softmax
        loss = F.cross_entropy(
            pred_logits,
            target_labels,
            weight=weight, # PyTorch 的 cross_entropy 支持样本权重
            reduction='none', # 先不 reduction，以便应用 loss_weight 和可能的样本 weight
            label_smoothing=self.label_smoothing
        )

        # 应用 reduction
        if self.reduction == 'mean':
            if weight is not None:
                 # 加权平均
                loss = (loss * weight).sum() / weight.sum()
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # 如果是 'none'，则不处理

        return self.loss_weight * loss

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram