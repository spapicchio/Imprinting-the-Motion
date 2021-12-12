import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, stackSize,train):
    Dataset = []
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'processed_frames2')

    for dir_user in sorted(os.listdir(root_dir)): 
      if train is False:
        if dir_user!='S2':
          continue
      elif dir_user=='S2':
        continue

      class_id = 0
      if not os.path.isdir(os.path.join(root_dir, dir_user)):
        continue
      dir = os.path.join(root_dir, dir_user)
      for target in sorted(os.listdir(dir)):
          if not os.path.isdir(os.path.join(dir, target)):
            continue
          dir1 = os.path.join(dir, target)
          insts = sorted(os.listdir(dir1))
          if insts != []:
               for inst in insts:
                if not os.path.isdir(os.path.join(dir1, inst)):
                  continue
                inst_dir = os.path.join(dir1, inst, 'rgb')
                numFrames = len(glob.glob1(inst_dir, '*.png'))
                if numFrames >= stackSize:
                   Dataset.append(inst_dir)
                   Labels.append(class_id)
                   NumFrames.append(numFrames)
          class_id += 1
    return Dataset, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png', flag_class=None, phase = 'train'):

        self.images, self.labels, self.numFrames = gen_split(root_dir, 5,train)
        self.st_1, self.st_rgb, _ = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt
        print('================ make dataset RGB ==============')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        temp = []
        self.st_1.randomize_parameters()

        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            img = self.st_1(img.convert('RGB'))
            img = self.st_rgb(img)
            inpSeq.append(img)
            temp.append(i)
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, temp, label




