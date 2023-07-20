from torch.utils.data import Dataset
import json
import numpy as np
from MultiviewX import MultiviewX
import torch
from PIL import Image
import cv2
import kornia
import torchvision.transforms as T 
from utils import get_gt



class Mutiview_dataloader(Dataset):
    def __init__(self,base, train=True, reID=False,
                 world_kernel_size=10, img_kernel_size=10,
                 train_ratio=0.9, top_k=100, semi_supervised=0.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.top_k = top_k
        self.root, self.num_cam, self.num_frame =base.root, base.num_cam, base.num_frame
        self.world_reduce, self.img_reduce = base.world_reduce, base.img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.semi_supervised = semi_supervised * train
        self.dropout = dropout
        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

    def __getitem__(self, index):
        img_path  =self.base.get_img_path(index)
        ann_path=self.base.get_ann_path(index)
        with open(ann_path,'r') as f:
            file = json.load(f) 
            bbox_dict= {}
            map_dict = {}
            for info in file:
                personID = str(info['personID'])
                positionID = self.base.get_worldgrid_from_pos(info['positionID'])
                views = info['views']
                templist=[]
                for camnum in range(self.num_cam): 
                    xmin = views[camnum]['xmin']
                    ymin = views[camnum]['ymin']
                    xmax = views[camnum]['xmax']
                    ymax = views[camnum]['ymax']
                    img_x_s, img_y_s = (xmin + xmax) / 2, ymin
                    img_w_s, img_h_s = (xmax - xmin), (ymax - ymin)
                    templist.append(np.array([img_x_s, img_y_s, img_w_s, img_h_s]))
                bbox_dict[personID]= templist
                map_dict[personID] = [positionID[0],positionID[1]]
        f.close()
        imgs_tensor,imgs_gt=[],[]
        img_pids = list(map(int,bbox_dict.keys()))
        for cam in range(len(img_path)):
            img = cv2.imread(img_path[cam])
            img = cv2.resize(img,[self.Rimg_shape[1]*8,self.Rimg_shape[0]*8])
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            imgs_tensor.append(img)
            x,y,w,h=[],[],[],[] #person_head_cam
            for key in bbox_dict.keys():
                params = bbox_dict[key]
                x.append(params[cam][0])
                y.append(params[cam][1])
                w.append(params[cam][2])
                h.append(params[cam][3])
            img_gt = get_gt(self.Rimg_shape, x, y, w, h, v_s=img_pids,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            imgs_gt.append(img_gt)
        imgs_tensor = torch.stack(imgs_tensor) 
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt])for key in imgs_gt[0]}
        x_map,y_map=[],[] #person_head_map
        for key_map in map_dict.keys():
            params_map = map_dict[key_map]
            x_map.append(params_map[0])
            y_map.append(params_map[1])
        world_gt = get_gt(self.Rworld_shape,x_map,y_map,v_s=img_pids,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        return imgs_tensor, imgs_gt, world_gt, index
    
    def __len__(self):
        return self.num_frame
    
if __name__ == '__main__':
    import torch
    data = MultiviewX('Data/MultiviewX')
    train_set = Mutiview_dataloader(data,True)
    imgs_tensor,imgs_gt,world_gt = train_set.__getitem__(0)