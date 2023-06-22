"""
A scaling layer makes adding the physical constraints easier.
Training using scaled data works better for learning, but the inputs need to be scaled and unscaled to preserve their
physical properties.
Adding a scaled layer makes pytorch use the underlying code to ensure this property
"""

import torch
import torch.nn as nn


class ScaleFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ipt, scale, shift):
        """
        Scales the input with the formula f = (ipt - shift) / scale. Corresponds to simple min-max-scaling
        :param ctx:
        :param ipt: Tensor, input
        :param scale:
        :param shift:
        :return:
        """
        ctx.save_for_backward(ipt, scale)
        return ipt * scale

    @staticmethod
    def backward(ctx, grad_output):
        ipt, scale = ctx.saved_variables
        return grad_output * scale, torch.mean(grad_output * ipt)


class ScaleLayer(nn.Module):
    def __init__(self, initial_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(initial_value))

    def forward(self, ipt):
        return ScaleFunc.apply(input, self.scale)
