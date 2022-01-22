import model
from dataset import Pair, center_crop

import os
import glob
import random
import cv2
import numpy as np
import time
import argparse

import torch
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--vis', action='store_true', help='whether to visualize')
parser.add_argument('--save', action='store_true', help='save video')

args = parser.parse_args()


def demo(data_path, model, save_video=False, vis=False):
    print("***Begin to tracking***")

    total = 0
    mean_fps = 0
    template_size = 25
    search_size = 65
    search_offset = int(search_size/2)
    center = search_offset + 1

    net = model

    transform = transforms.ToTensor()

    hanning = np.hanning(search_size)
    hanning = np.outer(hanning, hanning)

    img_path = sorted(glob.glob(os.path.join(data_path, '0*')))
    anno_path = sorted(glob.glob(os.path.join(data_path, 'labels/*')))
    anno_file = open(anno_path[0], encoding='gbk')
    anno = []
    score = 0

    for line in anno_file:
        anno.append(line.strip())

    anno_template = anno[1].split('\t')

    x_t = int(anno_template[3])
    y_t = int(anno_template[4])

    template = np.array(cv2.imread(img_path[0], 0), dtype=np.uint8)
    template = center_crop(template, [template_size, template_size], x_t, y_t)
    template = transform(template)
    template = template.cuda()
    template = torch.unsqueeze(template, 0)
    template = net.conv(template)
    self_attn_template = net.self_attn(template)

    if vis:
        cv2.namedWindow('demo', 0)
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f"./output/output.avi", fourcc, 30.0, (256, 256))

    for i in range(len(img_path)):
        time_start = time.time()
        o_search = np.array(cv2.imread(img_path[i], 0), dtype=np.uint8)

        search_region = center_crop(o_search, [search_size, search_size], x_t, y_t)

        search = transform(search_region)
        search = search.cuda()
        search = torch.unsqueeze(search, 0)
        search = net.conv(search)
        self_attn_search = net.self_attn(search)

        cross_attn_template = net.cross_attn(template, search)
        cross_attn_search = net.cross_attn(search, template)
        attn_template = torch.cat((self_attn_template, cross_attn_template), dim=1)
        attn_search = torch.cat((self_attn_search, cross_attn_search), dim=1)

        attn_template = net.adjust_attn(attn_template)
        attn_search = net.adjust_attn(attn_search)

        output = net.match_corr(attn_template, attn_search)

        output = torch.squeeze(output, 0)
        output = torch.squeeze(output, 0)

        prediction = output.cpu().detach().numpy()
        prediction -= np.min(prediction)
        prediction /= np.max(prediction)

        prediction = hanning * prediction
        position = np.unravel_index(np.argmax(prediction), prediction.shape)

        displace_x = position[1] - center + 1
        displace_y = position[0] - center + 1

        x_t += displace_x
        y_t += displace_y

        anno_search = anno[i+1].split('\t')
        x_s = int(anno_search[3])
        y_s = int(anno_search[4])

        distance = np.sqrt((x_t-x_s)**2 + (y_t-y_s)**2)
        if distance < 8:
            score += 1 - distance/8

        time_end = time.time()
        mean_fps += time_end - time_start

        if vis:
            o_search = cv2.cvtColor(o_search, cv2.COLOR_GRAY2BGR)

            cv2.rectangle(o_search, (x_t-7, y_t-7), (x_t+7, y_t+7), (0, 0, 255), 1)

            cv2.putText(o_search, 'Fps:%f' % (1 / (time_end - time_start)), (5, 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)

            cv2.imshow('demo', o_search)

        if save_video:
            out.write(o_search)
        cv2.waitKey(1)

    score = score / len(img_path)
    print(f'score: {score}')
    print(f'mean fps: {1 / (mean_fps/len(img_path))}')
    total += score
    return total


if __name__ == '__main__':
    data_path = './video'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load('./train_model/best.pkl')
    net.to(device)
    net.eval()

    total_score = demo(data_path, net, save_video=args.save,vis=args.vis)
    print(f'score: {total_score}')

    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total parameters: {total_num}, Trainable parameters: {trainable_num}')
