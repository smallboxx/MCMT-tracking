import numpy as np
import cv2
from PIL import Image
import torch

def get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    threeD2twoD = np.array([[1, 0, 0], [0, 1, 0], [0, 0, z], [0, 0, 1]])
    project_mat = intrinsic_mat @ extrinsic_mat @ threeD2twoD
    return project_mat

def get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    project_mat = np.linalg.inv(get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z))
    return project_mat


def array2heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return heatmap

class img_color_denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1])
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1])

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)
