from MultiviewX import MultiviewX
from Mutiviewdataloader import Mutiview_dataloader
from Mutiviewmodel import Mutiview_Model
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
import random
from utils import FocalLoss,RegCELoss,RegL1Loss
from utils import mvdet_decode,nms,evaluateDetection_py
import tqdm

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args):
    dataset   = MultiviewX(args.data)
    train_set = Mutiview_dataloader(dataset)
    train_loader = DataLoader(train_set,batch_size=args.batch_size, shuffle=True, pin_memory=True, worker_init_fn=seed_worker)
    model     = Mutiview_Model(train_set).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=400,epochs=10)
    mse_loss = torch.nn.MSELoss()
    focal_loss = FocalLoss()
    regress_loss = RegL1Loss()
    ce_loss = RegCELoss()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    for epoch in range(1, args.epochs + 1):
        losses = 0
        for batch_idx, (imgs_tensor,imgs_gt,world_gt,frame) in enumerate(train_loader):
            B, N = imgs_gt['heatmap'].shape[:2]
            (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(imgs_tensor.float().cuda())
            loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
            loss_w_off = regress_loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
            loss_img_hm = focal_loss(imgs_heatmap, imgs_gt['heatmap'])
            loss_img_off = regress_loss(imgs_offset, imgs_gt['reg_mask'][0], imgs_gt['idx'][0], imgs_gt['offset'][0])
            loss_img_wh = regress_loss(imgs_wh, imgs_gt['reg_mask'][0], imgs_gt['idx'][0], imgs_gt['wh'][0])
            w_loss = loss_w_hm + loss_w_off  # + self.id_ratio * loss_w_id
            img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.id_ratio * loss_img_id
            loss = w_loss + img_loss / N * 1.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if batch_idx + 1 == len(train_loader):
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, maxima: {world_heatmap.max():.3f}')
    if args.savelog:
        torch.save(model.state_dict(),'log/Multiview.pth')


def test(args):
    dataset      = MultiviewX(args.data)
    test_set     = Mutiview_dataloader(dataset)
    test_loader  = DataLoader(test_set,batch_size=args.batch_size, shuffle=False, pin_memory=True, worker_init_fn=seed_worker)
    model        = Mutiview_Model(test_set).cuda()
    mse_loss     = torch.nn.MSELoss()
    focal_loss   = FocalLoss()
    regress_loss = RegL1Loss()
    ce_loss      = RegCELoss()
    model.load_state_dict(torch.load('Multiview.pth'))
    model.eval()
    losses = 0
    cls    = 0.6
    res_list = []
    res_fpath = 'test.txt'
    gt_fpath  = 'Data/MultiviewX/gt.txt'
    for batch_idx, (imgs_tensor,imgs_gt,world_gt,frame) in enumerate(test_loader):
        if batch_idx == 1:
            break
        print(batch_idx,400)
        B, N = imgs_gt['heatmap'].shape[:2]
        for key in imgs_gt.keys():
            imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
        with torch.no_grad():
            (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(imgs_tensor.float().cuda())
            loss_w_hm = focal_loss(world_heatmap, world_gt['heatmap'])
            loss = loss_w_hm
            losses += loss.item()
            xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(), reduce=4)
            grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
            positions = grid_xy
            for b in range(B):
                ids = scores[b].squeeze() > cls
                pos, s = positions[b, ids], scores[b, ids, 0]
                res = torch.cat([torch.ones([len(s), 1]) * frame, pos], dim=1)
                ids, count = nms(pos, s, 20, np.inf)
                res = torch.cat([torch.ones([count, 1]) * frame[b], pos[ids[:count]]], dim=1)
                res_list.append(res)
    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])

    if args.savelog:
        if not os.path.exists(res_fpath):
            with open(res_fpath,'w') as f:
                f.write('')
        np.savetxt(res_fpath, res_list, '%d')

    if args.evaluate:
        recall, precision, moda, modp = evaluateDetection_py(os.path.abspath(res_fpath), os.path.abspath(gt_fpath))
        print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')

    if args.vis:
        dataset.vis_map(res_fpath, gt_fpath)

def run(args):
    if args.model == 'train':
        train(args)
        return
    if args.model == 'test':
        test(args)
        return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mutiview')
    parser.add_argument('--model', type=str, default='test', help='train or test')
    parser.add_argument('--data', type=str, default='Data/MultiviewX')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--savelog', type=bool, default=False)
    parser.add_argument('--vis', type=bool, default=True)
    args = parser.parse_args()

    run(args)



