import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import kornia
from utils import get_worldcoord_from_imgcoord_mat,resnet18,array2heatmap,img_color_denormalize,ConvWorldFeat
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def output_head(in_dim, feat_dim, out_dim):
    if feat_dim:
        fc = nn.Sequential(nn.Conv2d(in_dim, feat_dim, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(feat_dim, out_dim, 1))
    else:
        fc = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1))
    return fc


class Mutiview_Model(nn.Module):
    def __init__(self, dataset, base_dim = 512, bottleneck_dim=128, outfeat_dim=64, droupout=0.5, worldgrid_shape=[640, 1000],z=0):
        super(Mutiview_Model, self).__init__()
        self.worldgrid_shape = worldgrid_shape
        self.img_reduce = dataset.base.img_reduce
        self.Rworld_shape = list(map(lambda x: x // dataset.base.world_reduce, self.worldgrid_shape))
        world_zoom_mat = np.diag([dataset.world_reduce, dataset.base.world_reduce, 1])
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(
            dataset.base.worldcoord_from_worldgrid_mat @ world_zoom_mat @ dataset.base.world_indexing_from_xy_mat)
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(dataset.base.intrinsic_matrices[cam],
                                                                          dataset.base.extrinsic_matrices[cam],
                                                                          z / dataset.base.worldcoord_unit)
                                         for cam in range(dataset.base.num_cam)]
        self.proj_mats = torch.stack([torch.from_numpy(Rworldgrid_from_worldcoord_mat @
                                                       worldcoord_from_imgcoord_mats[cam])
                                      for cam in range(dataset.base.num_cam)])
        
        self.base = nn.Sequential(*list(resnet18(pretrained=True,replace_stride_with_dilation=[False, True, True]).children())[:-2])
        self.bottleneck = nn.Sequential(nn.Conv2d(base_dim, bottleneck_dim, 1), nn.Dropout2d(droupout))
        base_dim = bottleneck_dim
        self.img_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.img_offset = output_head(base_dim, outfeat_dim, 2)
        self.img_wh = output_head(base_dim, outfeat_dim, 2)

        self.world_head = ConvWorldFeat(dataset.base.num_cam, self.Rworld_shape, base_dim)
        self.world_heatmap = output_head(base_dim, outfeat_dim, 1)
        self.world_offset = output_head(base_dim, outfeat_dim, 2)

        # init
        self.img_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.img_offset)
        fill_fc_weights(self.img_wh)
        self.world_heatmap[-1].bias.data.fill_(-2.19)
        fill_fc_weights(self.world_offset)


    def forward(self, imgs_tensor,vis=False):
        B, N, C, H, W = imgs_tensor.shape
        imgs_tensor = imgs_tensor.view(B * N, C, H, W)
        imgcoord_from_Rimggrid_mat =torch.from_numpy(np.diag([self.img_reduce, self.img_reduce, 1])
                                                      ).view(1, 3, 3).repeat(B * N, 1, 1).float()
        # Rworldgrid(xy)_from_Rimggrid(xy)
        proj_mats = self.proj_mats.repeat(B, 1, 1, 1).view(B * N, 3, 3).float() @ imgcoord_from_Rimggrid_mat
        
        imgs_head = self.base(imgs_tensor)
        imgs_head = self.bottleneck(imgs_head)

        _, C, H, W = imgs_head.shape
        imgs_heatmap = self.img_heatmap(imgs_head)
        imgs_offset = self.img_offset(imgs_head)
        imgs_wh = self.img_wh(imgs_head)

        H, W = self.Rworld_shape
        world_head = kornia.geometry.warp_perspective(imgs_head.to(torch.float32), proj_mats.to(imgs_head.device),
                                             self.Rworld_shape, align_corners=False).view(B, N, C, H, W)
        if vis:
            for cam in range(N):
                heat_img = array2heatmap(torch.norm(imgs_head[cam * B].detach(), dim=0).cpu())
                plt.imshow(heat_img)
                plt.show()
                pro_img  = array2heatmap(torch.norm(world_head[0, cam].detach(), dim=0).cpu())
                plt.imshow(pro_img)
                plt.show()

        world_head = self.world_head(world_head)
        world_heatmap = self.world_heatmap(world_head)
        world_offset = self.world_offset(world_head)

        if vis:
            head_img = array2heatmap(torch.norm(world_head[0].detach(), dim=0).cpu())
            plt.imshow(head_img)
            plt.show()
            wheat_img = array2heatmap(torch.sigmoid(world_heatmap.detach())[0, 0].cpu())
            plt.imshow(wheat_img)
            plt.show()

        return (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh)
