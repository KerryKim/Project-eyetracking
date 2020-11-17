## import libraries
import os
import argparse
import random
import numpy as np

import cv2
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# self-class
from model import ITrackerModel
from dataset import *
from util import *


## parameters
mode = 'test'
ckpt_dir = './checkpoint'
data_dir = './data_test'

batch_size = 8
imSize = (224, 224)

# units in cm
screen_w = 5.58
screen_h = 10.45
screen_aspect = screen_w / screen_h
camera_l = 2.299
camera_t = 0.91
screen_t = 1.719
screen_l = 0.438
phone_w = 6.727
phone_h = 13.844
screen_from_camera = [screen_t - camera_t, screen_l - camera_l] # 그림파일의 빨간색 직선거리 [0.809, -1.861]


camera_coords_percentage = [camera_t / phone_h, camera_l / phone_w]

#iphone 8 screen w and screen height
screenW = 375
screenH = 667

phone_w_to_screen = phone_w / screen_w
phone_h_to_screen = phone_h / screen_h

camera_coords_percentage = [camera_t / phone_h, camera_l / phone_w]

#iphone 8 screen w and screen height
screenW = 375
screenH = 667

phone_w_to_screen = phone_w / screen_w
phone_h_to_screen = phone_h / screen_h


## render output
def render_gaze(full_image, camera_center, cm_to_px, output):
    xScreen = output[0]
    yScreen = output[1]
    pixelGaze = [round(camera_center[0] - yScreen * cm_to_px), round(camera_center[1] + xScreen * cm_to_px)]

    cv2.circle(full_image, (int(pixelGaze[1]), int(pixelGaze[0])), 30, (0, 0, 255), -1)


def render_gazes(index, img, output):
    full_image = np.ones((round(img.shape[0] * 2), round(img.shape[1] * 2), 3), dtype=np.uint8)  # 검은 배경 이미지

    full_image_center = [round(full_image.shape[0] * 0.2), round(full_image.shape[1] * .5)]
    camera_center = full_image_center
    cm_to_px = img.shape[0] * 1. / screen_h

    screen_from_camera_px = [round(screen_from_camera[0] * cm_to_px), round(screen_from_camera[1] * cm_to_px)]

    screen_start = [camera_center[0] + screen_from_camera_px[0], camera_center[1] + screen_from_camera_px[1]]

    full_image[screen_start[0]:screen_start[0] + img.shape[0], screen_start[1]:screen_start[1] + img.shape[1], :] = img[:,:,:]

    # bule circle is camera center
    cv2.circle(full_image, (camera_center[1], camera_center[0]), 30, (255, 0, 0), -1)  # 파랑색 점을 찍는다.

    if output is not None:
        render_gaze(full_image, camera_center, cm_to_px, output)

    plt.figure(figsize=(10, 10))
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

    # plt.imshow(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")
    plt.imsave(os.path.join('./test_result', 'input_%05d.jpg' % index), full_image)

## other fns
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()

def flatten(lst):
    result = []
    result = []
    for item in lst:
        result.extend(item)
    return result

## Network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = ITrackerModel().to(device)

## Optimizer
optim = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)


## TEST MODE
if mode == 'test':
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        dataset_test = Dataset(data_dir=data_dir, split='test', imSize=imSize)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)

        num_batch_test = np.ceil(num_data_test / batch_size)

        net.eval()
        pred = []

        for batch, input in enumerate(loader_test, 1):
            imFace, imEyeL, imEyeR, faceGrid = input.to(device)

            output = net(imFace, imEyeL, imEyeR, faceGrid)

            output = fn_tonumpy(output)
            output = flatten(output)
            pred.append(output)

            print("TEST: BATCH %04d / %04d" %
                  (batch, num_batch_test))

        result = './test_result'
        if not os.path.exists(result):
            os.makedirs(result)

        lst_data = os.listdir('./checkFace')

        for i in lst_data:
            img = cv2.imread(i)
            render_gazes(i, img, output[i])






