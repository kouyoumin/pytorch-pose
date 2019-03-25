from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math
import cv2

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *


class CmuHand(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path'] # root image folders
        self.jsonfile   = kwargs['anno_path']
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']

        if is_train:
            self.imglist = sorted(f for f in os.listdir(os.path.join(self.img_folder, 'train')) if f.endswith('.png'))
        else:
            self.imglist = sorted(f for f in os.listdir(os.path.join(self.img_folder, 'test')) if f.endswith('.png'))
        #print(self.imglist)
        self.imgs = []
        self.anno = []
        for filename in self.imglist:
            img = cv2.imread(os.path.join(self.img_folder, 'train', filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            annofile = filename[:-3]+'json'
            with open(os.path.join(self.img_folder, 'train', annofile), 'r') as f:
                anno = json.load(f)
                if anno is None:
                    continue
                self.imgs.append(os.path.join(self.img_folder, 'train', filename))
                self.anno.append(anno)

        assert(len(self.imgs)==len(self.anno))
        
        # create train/val split
        '''with open(self.jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train_list, self.valid_list = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid_list.append(idx)
            else:
                self.train_list.append(idx)'''
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = './data/cmuhand/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(1)
            std = torch.zeros(1)
            for img_path in self.imgs:
                #a = self.anno[index]
                #img_path = os.path.join(self.img_folder, self.imgs[index])
                img = load_image(img_path, mode='L') # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.imgs)
            std /= len(self.imgs)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %r' % (meanstd['mean']))
            print('    Std:  %r' % (meanstd['std']))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        a = self.anno[index]
        img_path = self.imgs[index]
        pts = torch.Tensor(a['hand_pts'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['hand_box_center'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        #if c[0] != -1:
        #    c[1] = c[1] + 15 * s
        #    s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='cmuhand')
                c[0] = img.size(2) - c[0]

            # Color
            #img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            #img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            #img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

            #Gamma
            gamma = random.uniform(3./4, 4./3)
            # Make distribution balanced 
            if random.random() < 0.5:
                gamma = 1/gamma
            img.pow_(1/gamma)

        # Prepare image and groundtruth map
        #print(img_path, c, s, r)
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index' : index, 'center' : c, 'scale' : s,
        'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.imgs)
        else:
            return len(self.imgs)


def cmuhand(**kwargs):
    return CmuHand(**kwargs)

cmuhand.njoints = 21  # ugly but works
