import torch.nn as nn
import torch


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, inp):
        if inp.size(1) != 784 and inp.size(1) != 2352:
            inp.data = Binarize(inp.data)
        else:
            # inp.data = inp.data.mul(255)
            inp.data = inp.data.sign().mul(2).sub(1).sign()  # Binarize(inp.data)

        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = nn.functional.linear(inp, self.weight)

        """
        bias
        """
        if not self.bias is None:
            # if not hasattr(self.bias, 'org'):
            #     self.bias.org = self.bias.data.clone()
            out = out.cuda() + self.bias.view(1, -1).expand_as(out)
        return out


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign().mul(2).add(1).sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)
