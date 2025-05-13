import torch
import Model_Integration

class VanillaConv2D(torch.nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask.contiguous().float().cuda()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        for b in range(B):
            custom_conv.vanilla_convolve(x[b], self.mask, out[b])
        return out


class PTXConv2D(torch.nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.mask = mask.contiguous().float().cuda()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        for b in range(B):
            custom_conv.ptx_convolve(x[b], self.mask, out[b])
        return out

