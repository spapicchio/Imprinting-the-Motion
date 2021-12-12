import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random
from spatial_transforms import (Compose, ToTensor, Scale)


def gen_split(root_dir, stackSize, train):
    
    Dataset_rgb = []
    Dataset_mmap=[]
    Labels = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'processed_frames2')

    for dir_user in sorted(os.listdir(root_dir)): 
      
      if train is False: #validation/test
        if dir_user!='S2':
          continue
      elif dir_user=='S2':#training
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
                inst_dir_rgb = os.path.join(dir1, inst, 'rgb')
                inst_dir_mmap = os.path.join(dir1, inst, 'mmaps')
                numFrames = len(glob.glob1(inst_dir_rgb, '*.png'))
                if numFrames >= stackSize:
                   Dataset_rgb.append(inst_dir_rgb)
                   Dataset_mmap.append(inst_dir_mmap)
                   Labels.append(class_id)
                   NumFrames.append(numFrames)
          class_id += 1
    return Dataset_rgb, Dataset_mmap, Labels, NumFrames

def k_largest(iterable, k=7):
    
    result = [] # O(k) space
    vals=[]
    
    for i in range(k):
        x=iterable[i]
        result.append((x,i))
        vals.append(torch.var(x))
    
    r_min= min(vals)
    for i,elem in enumerate(iterable[k:]):
        
        x=torch.norm(elem)
        if x > r_min:
            result.pop(vals.index(r_min))  # O(n*k) time
            vals.remove(r_min)
            result.append((elem, i+k))  
            vals.append(torch.var(elem))
            r_min= min(vals)
        
    return result # [(element, index), ...]

def selectDistant(l,k=7):
  sublen=3
  frames=[l[0]]
  i=0
  end=False
  if(len(l))==k:   #terminazione
    return l
    
  while(i<=len(l)):
      maxV=0
      maxI=i
        
      for n in range(1,sublen): 
          j=n+i
          if(j>=len(l)):
              end=True
              break
          dist=torch.dist(l[i],l[j])
          if dist>maxV:
            maxV=dist
            maxI=j 
        
      frames.append(l[maxI])
      i=maxI
      if len(frames)==k:
          end=True
      if end==True:
          i=len(l)+1       
    
  f=len(frames)
    
  if (f>k):
      frames=selectDistant(frames)   
  elif f<k:
      p=k-f
      while n!=0:
          frames.append(l[-p])
          p=p-1
            
  return frames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, seqLen=7,
train=True, mulSeg=False, numSeg=1, fmt='.png', flag_class = True, phase='train'):
        self.images, self.mmap, self.labels, self.numFrames = gen_split(root_dir, 5, train)
        self.st_1, self.st_rgb, self.st_mmaps = spatial_transform
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt
        self.flag_class = flag_class #ms_class
        print('=================== make_dataset_self_supervised ===============')
        print(flag_class)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        vid_name = self.images[idx]
        vid_name_mmap = self.mmap[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        inpSeq_mmap = []

        self.st_1.randomize_parameters()


        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            img = self.st_1(img.convert('RGB'))
            img = self.st_rgb(img)
            inpSeq.append(img)
            

            for j in range(numFrame - 1):

              fl_name = vid_name_mmap + '/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
              if os.path.isfile(fl_name):
                break
              
              fl_name = vid_name_mmap + '/' + 'map' + str(int(np.floor(i + j))).zfill(4) + self.fmt
              if os.path.isfile(fl_name):
                break
              
              fl_name = vid_name_mmap + '/' + 'map' + str(int(np.floor(i - j))).zfill(4) + self.fmt
              if os.path.isfile(fl_name):
                break
                
            else: raise Exception(f"No elements in the folder:{vid_name_mmap + '/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt}")

            img = Image.open(fl_name)
            img = self.st_1(img)
            img = self.st_mmaps(img)

            if self.flag_class: #classification flag #True or False
                img = (img > 0.2).long() #hyperparameter?
            

            img = torch.squeeze(img) #from 1,7,7 to 7,7
           
            
            #print()
            #print(f'img size = {torch.numel(img)}')
            #print(f'img no zero = {torch.count_nonzero(img)}')
            #print()
            inpSeq_mmap.append(img)
        
        #inpSeq_index = k_largest(inpSeq, 7)
        #my_inpSeq = []
        #my_mmap = []
        #for x_i in inpSeq_index:
        #    i = x_i[1] #take index
        #    x = x_i[0] #take frame i 
        #    
        #    my_inpSeq.append(x)
        #    my_mmap.append(inpSeq_mmap[i])
            
        #inpSeq = torch.stack(my_inpSeq, 0)
        #inpSeq_mmap=torch.stack(my_mmap, 0)
        #return inpSeq, inpSeq_mmap, label
           


        #my_inpSeq = selectDistant(inpSeq, 7)
        #my_mmap = []

        #print(type(inpSeq_mmap)) 
        #for i,x in enumerate(my_inpSeq):
        #  print(i,x.size())
        #  my_mmap.append(inpSeq_mmap[inpSeq.index(x)])

        #print('============ fatto ==============')
        #inpSeq = torch.stack(my_inpSeq, 0)
        #inpSeq_mmap=torch.stack(my_mmap, 0)
        #return inpSeq, inpSeq_mmap, label
        
        #inpSeq = torch.stack(my_inpSeq, 0)
        #inpSeq_mmap=torch.stack(my_mmap, 0)
        #return inpSeq, inpSeq_mmap, label

        inpSeq = torch.stack(inpSeq, 0)
        inpSeq_mmap=torch.stack(inpSeq_mmap, 0)
        return inpSeq, inpSeq_mmap, label
        
        