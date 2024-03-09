import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import cv2

SCENES = ['01_', '02_', '03_', '04_', '05_', '06_',
          '07_', '08_', '09_', '10_', '11_', '12_', '13_']


class STCDataset(Dataset):
    def __init__(self, root_path=r'/data/dataset/AnomalyDetection/shanghaitech', scene='01_',
                 isTrain=True, resize=256, cropsize=224):
        self.root_path = root_path
        assert scene in SCENES
        self.scene = scene
        self.isTrain = isTrain
        self.train_path = os.path.join(root_path, 'training')
        self.train_videos_path = os.path.join(self.train_path, 'videos')
        self.train_frames_path = os.path.join(self.train_path, 'frames')
        self.test_path = os.path.join(root_path, 'testing')
        self.resize = resize
        self.cropsize = cropsize

        self.inter = 5
        # load dataset
        self.x, self.y, self.mask = self.load_dataset()

        # set transforms
        self.transform_x = T.Compose([T.Resize(self.resize, Image.ANTIALIAS),
                                      T.CenterCrop(self.cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(self.resize, Image.NEAREST),
                                         T.CenterCrop(self.cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):

        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        # mask = np.uint8(np.where(mask != 0, 1, 0))
        if mask is not None:
            mask = Image.fromarray(mask)
            mask = self.transform_mask(mask)
        else:
            mask = np.zeros([1, x.shape[1], x.shape[2]])

        return x, y, mask

    def __len__(self):
        return len(self.x)

    # frame_mask: n
    # pixel_mask: n * h * w
    def load_dataset(self):
        x, y, mask = [], [], []
        if self.isTrain:
            splited_videos = sorted(os.listdir(self.train_frames_path))
            for splited_video in splited_videos:
                # print("splited_video: ", splited_video)
                if self.scene in splited_video:
                    # print("scene: ", self.scene)
                    splited_video_path = os.path.join(self.train_frames_path, splited_video)
                    # print("splited_video_path: ", splited_video_path)
                    # frames = sorted(os.listdir(splited_video_path))
                    img_fpath_list = sorted([os.path.join(splited_video_path, f)
                                             for f in os.listdir(splited_video_path)
                                             if f.endswith('.jpg')])
                    # print("img_fpath_list: ", img_fpath_list)
                    x.extend(img_fpath_list)
                    # y.extend()
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
        else:
            # test
            # test_folders = [os.path.join(self.test_path, f)
            #                 for f in os.listdir(self.test_path)
            #                 if os.path.isdir(os.path.join(self.test_path, f))]
            test_folders = os.listdir(self.test_path)
            for test_folder in test_folders:
                test_path = os.path.join(self.test_path, test_folder)
                if not os.path.isdir(test_path):
                    continue
                if test_folder == 'frames':
                    # x
                    frames = sorted(os.listdir(test_path))
                    for frame in frames:
                        if self.scene in frame:
                            splited_video_path = os.path.join(test_path, frame)
                            img_fpath_list = sorted([os.path.join(splited_video_path, f)
                                                     for f in os.listdir(splited_video_path)
                                                     if f.endswith('.jpg')])
                            x.extend(img_fpath_list)
                elif 'frame_mask' in test_path:
                    # y
                    y_fpath_list = sorted([os.path.join(test_path, f)
                                           for f in os.listdir(test_path)
                                           if f.endswith('.npy')])
                    for f in y_fpath_list:
                        if self.scene in f:
                            y.extend(np.load(f))
                elif 'pixel_mask' in test_path:
                    # mask
                    mask_fpath_list = sorted([os.path.join(test_path, f)
                                              for f in os.listdir(test_path)
                                              if f.endswith('.npy')])
                    for f in mask_fpath_list:
                        if self.scene in f:
                            mask.extend(np.load(f))
            x, y, mask = self.samples(x, y, mask)
        return x, y, mask

    def samples(self, x, y, mask):
        assert len(x) == len(y)
        length = len(x)
        new_x = []
        new_y = []
        new_mask = []
        for i in range(length):
            if i % self.inter == 0:
                new_x.append(x[i])
                new_y.append(y[i])
                new_mask.append(mask[i])
        return new_x, new_y, new_mask


if __name__ == '__main__':
    dataset = STCDataset(isTrain=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (x, y, mask) in enumerate(dataloader):
        print('')


