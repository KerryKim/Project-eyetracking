## import libraries
import os
import os.path
import numpy as np
import scipy.io as sio

import torch
import torch.utils.data as data

from PIL import Image
from torchvision import transforms


## mean images for standardization
def get_mean_image(data_dir, file_name):
    image_mean = np.array(sio.loadmat(data_dir + file_name)['image_mean'])
    image_mean = image_mean.reshape(3, 224, 224)
    return image_mean.mean(1).mean(1)


## Transform
class ToPILImage(object):
    def __call__(self, input):
        input = transforms.ToPILImage()(input)

        return input


class ToNumpy(object):
    def __call__(self, input):
        return np.array(input)


class SubtractMean(object):
    def __init__(self, meanImg):
        self.meanImg = meanImg

    def __call__(self, input):
        input = input - self.meanImg
        return input


class ToTensor(object):
    def __call__(self, input):
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        input = np.moveaxis(input[:, :, :], -1, 0)
        input = torch.from_numpy(input / 255)
        input = input.unsqueeze(0)

        return input


## define dataset
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, imSize=(224,224)):
        self.data_dir = data_dir
        self.split = split
        self.imSize = imSize

        metaFile = os.path.join(data_dir, 'metadata.mat')
        self.metadata = sio.loadmat(metaFile, squeeze_me=True, struct_as_record=False)
        self.faceMean = get_mean_image(data_dir=self.data_dir, file_name='mean_face_224.mat')
        self.eyeLeftMean = get_mean_image(data_dir=self.data_dir,file_name='mean_left_224.mat')
        self.eyeRightMean = get_mean_image(data_dir=self.data_dir,file_name='mean_right_224.mat')

        if split == 'test':
            self.transformFace = transforms.Compose(
                [ToPILImage(), transforms.Resize((224, 224)), SubtractMean(meanImg=self.faceMean), ToNumpy(), ToTensor()])
            self.transformEyeL = transforms.Compose(
                [ToPILImage(), transforms.Resize((224, 224)), SubtractMean(meanImg=self.eyeLeftMean), ToNumpy(), ToTensor()])
            self.transformEyeR = transforms.Compose(
                [ToPILImage(), transforms.Resize((224, 224)), SubtractMean(meanImg=self.eyeRightMean), ToNumpy(), ToTensor()])

        path = self.data_dir + '/data_test/appleFace'
        lst_input = os.listdir(path)
        lst_input.sort()

        self.lst_input = lst_input

     # dataset length
    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        # input (dictionary type only)
        imFace_dir = os.path.join(self.data_dir, 'appleFace/%05d.jpg' % (index))
        imEyeL_dir = os.path.join(self.data_dir, 'appleLeftEye/%05d.jpg' % (index))
        imEyeR_dir = os.path.join(self.data_dir, 'appleRightEye/%05d.jpg' % (index))
        faceGrid_dir = os.path.join(self.data_dir, 'appleGrid/%05d.jpg' % (index))

        imFace = Image.open(imFace_dir).convert('RGB')
        imEyeL = Image.open(imEyeL_dir).convert('RGB')
        imEyeR = Image.open(imEyeR_dir).convert('RGB')

        faceGrid = np.load(faceGrid_dir)
        faceGrid = faceGrid.reshape(1, 625, 1, 1)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)
        faceGrid = torch.FloatTensor(faceGrid)

        input = {'imFace': imFace, 'imEyeL': imEyeL, 'imEyeR' : imEyeR, 'faceGrid' : faceGrid}

        return input