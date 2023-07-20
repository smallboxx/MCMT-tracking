import os
import numpy as np
import cv2
import re
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import json
import tqdm

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']


class MultiviewX(Dataset):
    def __init__(self, root):
        super().__init__()
        # MultiviewX has xy-indexing: H*W=640*1000, thus x is \in [0,1000), y \in [0,640)
        # MultiviewX has consistent unit: meter (m) for calibration & pos annotation
        self.root = root
        self.img_shape, self.worldgrid_shape = [1080, 1920], [640, 1000]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 6, 400
        # world x,y correspond to w,h
        self.indexing = 'xy'
        self.world_indexing_from_xy_mat = np.eye(3)
        self.world_indexing_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # image is in xy indexing by default
        self.img_xy_from_xy_mat = np.eye(3)
        self.img_xy_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.world_reduce = 4
        self.img_reduce   = 16
        # unit in meters
        self.worldcoord_unit = 1
        self.worldcoord_from_worldgrid_mat = np.array([[0.025, 0, 0], [0, 0.025, 0], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices, self.distortion_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_img_path(self, index):
        img_paths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            img_paths[cam] = os.path.join(self.root, 'Image_subsets', camera_folder, str(index).zfill(4)+'.png')
        return img_paths
    
    def get_ann_path(self, index):
        annotation_folder = os.path.join(self.root, 'annotations_positions')
        annotation_path = os.path.join(annotation_folder,str(index).zfill(5)+'.json')
        return annotation_path
    
    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return grid_x + grid_y * 1000

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord
        grid_x = coord_x * 40
        grid_y = coord_y * 40
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid
        coord_x = grid_x / 40
        coord_y = grid_y / 40
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                      intrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        distortion_matrix = fp_calibration.getNode('distortion_coefficients').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                      extrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix, distortion_matrix

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam

    def visualization(self):
        if not os.path.exists('visulization'):
            os.makedirs('visulization')
        filepath = self.get_image_fpaths(det_frame=400,refrence_frame=400)
        annotation_path = self.get_annotation_fpaths(det_frame=400,refrence_frame=400)
        video = cv2.VideoWriter('visulization/visulization.avi', cv2.VideoWriter_fourcc(*"MJPG"),3,(1580, 1060))
        for index in tqdm.tqdm(list(annotation_path.keys())):
            img_comb = np.zeros([1060, 1580, 3]).astype('uint8')
            map_res = np.ones([640, 1000, 3]).astype('uint8')
            map_res = np.uint8(255 * map_res)
            cam = [cv2.imread(filepath[i][index]) for i in range(6)]
            with open(annotation_path[index],'r') as f:
                file = json.load(f) 
                for info in file:
                    personID = info['personID']
                    np.random.seed(personID)
                    color = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
                    positionID = self.get_worldgrid_from_pos(info['positionID'])
                    cv2.circle(map_res,(positionID[0],positionID[1]),7,color,-1)
                    views = info['views']
                    for camview in views: 
                        viewNum = camview['viewNum']
                        # print(personID,positionID,camview)
                        if camview['xmin'] == -1:
                            continue
                        cv2.rectangle(cam[viewNum],(camview['xmin'],camview['ymin']),(camview['xmax'],camview['ymax']),color,5)
                        cv2.putText(cam[viewNum],str(personID),(camview['xmin'],camview['ymin']),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            map_res = cv2.resize(map_res,(750,480))
            map_res = cv2.putText(map_res, 'Ground Plane', (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (87, 59, 233), 2, cv2.LINE_AA)
            img_comb[580:580 + 480, 500:500 + 750] = map_res
            for id in range(6):
                img = cv2.resize(cam[id],(480,270))
                img = cv2.putText(img, f'Camera {id + 1}', (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (87, 59, 233), 2, cv2.LINE_AA)
                i, j = id // 3, id % 3
                img_comb[i * 290:i * 290 + 270, j * 500:j * 500 + 480] = img
            video.write(img_comb)
            pass
        video.release()
        return

    def vis_map(self, res_fpath, gt_fpath):
        if not os.path.exists('visulization'):
            os.makedirs('visulization')
        gtRaw = np.loadtxt(gt_fpath)
        detRaw = np.loadtxt(res_fpath)
        frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
        print(gtRaw,detRaw,frames)
        video = cv2.VideoWriter('visulization/test.avi', cv2.VideoWriter_fourcc(*"MJPG"),3,(1580, 1060))
        return

def test():
    dataset = MultiviewX('Data/MultiviewX')
    dataset.visualization()



if __name__ == '__main__':
    test()

