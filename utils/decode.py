import torch
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

def nms(points, scores, dist_thres=50 / 2.5, top_k=50):
    assert points.shape[0] == scores.shape[0], 'make sure same points and scores have the same size'
    keep = torch.zeros_like(scores).long()
    if points.numel() == 0:
        return keep, 0
    v, indices = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    top_k = min(top_k, len(indices))
    indices = indices[-top_k:]  # indices of the top-k largest vals

    # keep = torch.Tensor()
    count = 0
    while indices.numel() > 0:
        idx = indices[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = idx
        count += 1
        if indices.numel() == 1:
            break
        indices = indices[:-1]  # remove kept element from view
        target_point = points[idx, :]
        # load bboxes of next highest vals
        remaining_points = points[indices, :]
        dists = torch.norm(target_point - remaining_points, dim=1)  # store result in distances
        # keep only elements with an dists > dist_thres
        indices = indices[dists > dist_thres]
    return keep, count

def mvdet_decode(scoremap, offset=None, reduce=4):
    B, C, H, W = scoremap.shape
    # scoremap = _nms(scoremap)

    xy = torch.nonzero(torch.ones_like(scoremap[:, 0])).view([B, H * W, 3])[:, :, [2, 1]].float()
    if offset is not None:
        offset = offset.permute(0, 2, 3, 1).reshape(B, H * W, 2)
        xy = xy + offset
    else:
        xy = xy + 0.5
    xy *= reduce
    scores = scoremap.permute(0, 2, 3, 1).reshape(B, H * W, 1)

    return torch.cat([xy, scores], dim=2)


def getDistance(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

def CLEAR_MOD_HUN(gt, det):
    td = 50 / 2.5  # distance threshold

    F = int(max(gt[:, 0])) + 1
    N = int(max(det[:, 1])) + 1
    Fgt = int(max(gt[:, 0])) + 1
    Ngt = int(max(gt[:, 1])) + 1

    M = np.zeros((F, Ngt))

    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))

    d = np.zeros((F, Ngt))
    distances = np.inf * np.ones((F, Ngt))

    for t in range(1, F + 1):
        GTsInFrames = np.where(gt[:, 0] == t - 1)
        DetsInFrames = np.where(det[:, 0] == t - 1)
        GTsInFrame = GTsInFrames[0]
        DetsInFrame = DetsInFrames[0]
        GTsInFrame = np.reshape(GTsInFrame, (1, GTsInFrame.shape[0]))
        DetsInFrame = np.reshape(DetsInFrame, (1, DetsInFrame.shape[0]))

        Ngtt = GTsInFrame.shape[1]
        Nt = DetsInFrame.shape[1]
        g[0, t - 1] = Ngtt

        if GTsInFrame is not None and DetsInFrame is not None:
            dist = np.inf * np.ones((Ngtt, Nt))
            for o in range(1, Ngtt + 1):
                GT = gt[GTsInFrame[0][o - 1]][2:4]
                for e in range(1, Nt + 1):
                    E = det[DetsInFrame[0][e - 1]][2:4]
                    dist[o - 1, e - 1] = getDistance(GT[0], GT[1], E[0], E[1])
            tmpai = dist
            tmpai = np.array(tmpai)

            # Please notice that the price/distance of are set to 100000 instead of np.inf, since the Hungarian Algorithm implemented in
            # sklearn will suffer from long calculation time if we use np.inf.
            tmpai[tmpai > td] = 1e6
            if not tmpai.all() == 1e6:
                HUN_res = np.array(linear_sum_assignment(tmpai)).T
                HUN_res = HUN_res[tmpai[HUN_res[:, 0], HUN_res[:, 1]] < td]
                u, v = HUN_res[HUN_res[:, 1].argsort()].T
                for mmm in range(1, len(u) + 1):
                    M[t - 1, u[mmm - 1]] = v[mmm - 1] + 1
        curdetected, = np.where(M[t - 1, :])

        c[0][t - 1] = curdetected.shape[0]
        for ct in curdetected:
            eid = M[t - 1, ct] - 1
            gtX = gt[GTsInFrame[0][ct], 2]

            gtY = gt[GTsInFrame[0][ct], 3]

            stX = det[DetsInFrame[0][int(eid)], 2]
            stY = det[DetsInFrame[0][int(eid)], 3]

            distances[t - 1, ct] = getDistance(gtX, gtY, stX, stY)
        fp[0][t - 1] = Nt - c[0][t - 1]
        m[0][t - 1] = g[0][t - 1] - c[0][t - 1]

    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 if sum(
        1 - distances[distances < td] / td) / np.sum(c) * 100 > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 if (1 - (
            (np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 if np.sum(c) / np.sum(g) * 100 > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 if np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 > 0 else 0

    return recall, precision, MODA, MODP


def evaluateDetection_py(res_fpath, gt_fpath):
    gtRaw = np.loadtxt(gt_fpath)
    detRaw = np.loadtxt(res_fpath)
    frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
    frame_ctr = 0
    gt_flag = True
    det_flag = True

    gtAllMatrix = 0
    detAllMatrix = 0
    if detRaw is None or detRaw.shape[0] == 0:
        MODP, MODA, recall, precision = 0, 0, 0, 0
        return MODP, MODA, recall, precision

    for t in frames:
        idxs = np.where(gtRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in gtRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in gtRaw[idx, 2]])

        if gt_flag:
            gtAllMatrix = tmp_arr
            gt_flag = False
        else:
            gtAllMatrix = np.concatenate((gtAllMatrix, tmp_arr), axis=0)
        idxs = np.where(detRaw[:, 0] == t)
        idx = idxs[0]
        idx_len = len(idx)
        tmp_arr = np.zeros(shape=(idx_len, 4))
        tmp_arr[:, 0] = np.array([frame_ctr for n in range(idx_len)])
        tmp_arr[:, 1] = np.array([i for i in range(idx_len)])
        tmp_arr[:, 2] = np.array([j for j in detRaw[idx, 1]])
        tmp_arr[:, 3] = np.array([k for k in detRaw[idx, 2]])

        if det_flag:
            detAllMatrix = tmp_arr
            det_flag = False
        else:
            detAllMatrix = np.concatenate((detAllMatrix, tmp_arr), axis=0)
        frame_ctr += 1
    recall, precision, MODA, MODP = CLEAR_MOD_HUN(gtAllMatrix, detAllMatrix)
    return recall, precision, MODA, MODP
