import os
import glob
import random

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Pair(Dataset):
    def __init__(self, data_path, transforms=None, rand_choice=True, gary=True,
                 img_size=256, search_size=63, template_size=25, r_pos=7):
        self.data_path = data_path
        self.transforms = transforms
        self.gary = gary
        self.rand_choice = rand_choice
        self.seq_path = sorted(glob.glob(os.path.join(self.data_path, 'data*')))
        self.anno_path = sorted(glob.glob(os.path.join(self.data_path, 'labels/*')))

        self.img_size = img_size
        self.template_size = template_size
        self.search_size = search_size
        self.scoreSize = search_size
        self.r_pos = r_pos

        self.pairs_per_video = 25
        self.frame_range = 100

        n = len(self.seq_path)
        self.indices = np.arange(0, n, dtype=int)
        self.indices = np.tile(self.indices, self.pairs_per_video)

    def __getitem__(self, index):
        if self.rand_choice:
            index = np.random.choice(self.indices)

        img_path = sorted(glob.glob(os.path.join(self.seq_path[index], '*')))

        anno_file = open(self.anno_path[index], encoding='gbk')
        anno = []
        right_data = True

        for line in anno_file:
            anno.append(line.strip())

        while right_data:
            rand_template, rand_search = self.sample_pair(len(img_path))
            anno_template = anno[rand_template + 1].split('\t')
            anno_search = anno[rand_search + 1].split('\t')
            if int(anno_template[1]) != 0 and int(anno_search[1]) != 0:
                right_data = False

        if self.gary:
            template_o = np.array(cv2.imread(img_path[rand_template], 0), dtype=np.uint8)
            search_o = np.array(cv2.imread(img_path[rand_search], 0), dtype=np.uint8)
        else:
            template_o = np.array(cv2.imread(img_path[rand_template]), dtype=np.uint8)
            search_o = np.array(cv2.imread(img_path[rand_search]), dtype=np.uint8)

        x_t = int(anno_template[3])
        y_t = int(anno_template[4])

        template = center_crop(template_o, [self.template_size, self.template_size], x_t, y_t)

        x_s = int(anno_search[3])
        y_s = int(anno_search[4])

        search = center_crop(search_o, [self.search_size, self.search_size], x_s, y_s)

        label, weight = self.create_labels()

        # print(index, rand_template, rand_search)
        # print(template_o.shape, search_o.shape)
        # cv2.circle(template_o, (x_t, y_t), 4, 255, 1)
        # cv2.circle(search_o, (x_s, y_s), 4, 255, 1)
        # cv2.imshow('template_o', template_o)
        # cv2.imshow('search_o', search_o)
        # cv2.imshow('template', template)
        # cv2.imshow('search', search)
        # cv2.waitKey(0)

        template = self.transforms(template)
        search = self.transforms(search)
        label = torch.FloatTensor(label)
        weight = torch.FloatTensor(weight)
        template = torch.FloatTensor(template)
        search = torch.FloatTensor(search)

        return template, search, label, weight

    def sample_pair(self, n):
        rand_z = np.random.randint(n)  # select a image randomly as z(template)
        if self.frame_range == 0:
            return rand_z, rand_z
        possible_x = np.arange(rand_z - self.frame_range,
                               rand_z + self.frame_range)  # get possible search(x) according to frame_range
        possible_x = np.intersect1d(possible_x, np.arange(n))  # remove impossible x(search)
        possible_x = possible_x[possible_x != rand_z]  # z(template) and x(search) cannot be same
        rand_x = np.random.choice(possible_x)  # select x from possible_x randomly
        return rand_z, rand_x

    def __len__(self):
        return len(self.indices)

    def create_labels(self):
        labels = self.create_logisticloss_new_labels()
        weights = np.zeros_like(labels)

        # cv2.imshow('labels', labels)
        # cv2.waitKey(0)

        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        # weights *= pos_num + neg_num

        labels = labels[np.newaxis, :]
        weights = weights[np.newaxis, :]

        return labels, weights

    def create_logisticloss_labels(self):
        label_sz = self.scoreSize
        label_center = self.scoreSize / 2
        r_pos = self.r_pos
        labels = np.zeros((label_sz, label_sz))

        for r in range(label_sz):
            for c in range(label_sz):
                if label_center - r_pos / 2 <= r < label_center + r_pos / 2 \
                        and label_center - r_pos / 2 <= c < label_center + r_pos / 2:
                    labels[r, c] = 1
                else:
                    labels[r, c] = 0
        return labels

    def create_logisticloss_new_labels(self):
        label_sz = self.scoreSize
        label_center = int(self.scoreSize / 2) + 1
        r_pos = self.r_pos
        labels = np.zeros((label_sz, label_sz))

        for r in range(label_sz):
            for c in range(label_sz):
                distance = np.sqrt((r - label_center) ** 2 + (c - label_center) ** 2)
                if distance <= r_pos:
                    labels[c, r] = 1
                else:
                    labels[c, r] = 0

        return labels


def center_crop(sample, size, cx=None, cy=None):
    shape = sample.shape[:2]

    if not cx or not cy:
        cy, cx = (shape[0] - 1) // 2, (shape[1] - 1) // 2

    ymin, xmin = cy - size[0] // 2, cx - size[1] // 2
    ymax, xmax = cy + size[0] // 2 + size[0] % 2, cx + size[1] // 2 + size[1] % 2
    left = right = top = bottom = 0
    im_h, im_w = shape
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > im_w:
        right = int(xmax - im_w)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > im_h:
        bottom = int(ymax - im_h)

    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = sample[ymin:ymax, xmin:xmax]

    if left != 0 or right != 0 or top != 0 or bottom != 0:
        im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return im_patch

