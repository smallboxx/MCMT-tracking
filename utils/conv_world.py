import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

def create_coord_map(img_size, with_r=False):
    H, W = img_size
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
    grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
    ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    if with_r:
        grid_r = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
        ret = torch.cat([ret, grid_r], dim=1)
    return ret


class ConvWorldFeat(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, stride=2, reduction=None):
        super(ConvWorldFeat, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 3, stride, 1), nn.ReLU(), )
        self.coord_map = create_coord_map(np.array(Rworld_shape) // stride)
        self.reduction = reduction
        if self.reduction is None:
            combined_input_dim = base_dim * num_cam + 2
        elif self.reduction == 'sum':
            combined_input_dim = base_dim + 2
        else:
            raise Exception
        self.world_feat = nn.Sequential(nn.Conv2d(combined_input_dim, hidden_dim, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), nn.ReLU(), )
        self.upsample = nn.Sequential(nn.Upsample(Rworld_shape, mode='bilinear', align_corners=False),
                                      nn.Conv2d(hidden_dim, base_dim, 3, 1, 1), nn.ReLU(), )

    def forward(self, x, visualize=False):
        B, N, C, H, W = x.shape
        x = self.downsample(x.view(B * N, C, H, W))
        _, _, H, W = x.shape
        if self.reduction is None:
            x = x.view(B, N * C, H, W)
        elif self.reduction == 'sum':
            x = x.sum(dim=1)
        else:
            raise Exception
        x = torch.cat([x, self.coord_map.repeat([B, 1, 1, 1]).to(x.device)], 1)
        x = self.world_feat(x)
        x = self.upsample(x)
        return x
