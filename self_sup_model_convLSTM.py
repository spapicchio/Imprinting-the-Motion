import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *


class selfSupModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, list_flags = (True, False, False, False)):
        super(selfSupModel, self).__init__()
        
        #set the flags
        spatial_attention, motion_segmentation, variation, classification = list_flags  
        self.spatial_attention = spatial_attention
        self.motion_segmentation = motion_segmentation
        self.variation = variation
        self.classification = classification
        
        #building the architecture
        self.num_classes = num_classes
       
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        
        self.avgpool = nn.AvgPool2d(7)
        
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        
        if motion_segmentation:
            self.relu_ms = nn.ReLU()
            self.conv100 = nn.Conv2d(in_channels=512,out_channels=100,kernel_size=1,padding=0)
            
            if classification:
              self.fc1=nn.Linear(in_features=100*7*7,out_features=2*7*7)
            else:
              self.fc1=nn.Linear(in_features=100*7*7,out_features=7*7)
            torch.nn.init.xavier_normal_(self.conv100.weight)
            torch.nn.init.constant_(self.conv100.bias, 0)
            torch.nn.init.xavier_normal_(self.fc1.weight)
            torch.nn.init.constant_(self.fc1.bias, 0)
            print('================ ms added ===============' )
        
        if variation:

                        
            self.betaConv = nn.Sequential(nn.Conv2d(100, 64, kernel_size=1, padding=0), nn.ReLU(),
                                          nn.Conv2d(64, 64, kernel_size=1, padding=0))
                                                      
            self.gammaConv = nn.Sequential(nn.Conv2d(100, 64, kernel_size=1, padding=0), nn.ReLU(),
                                           nn.Conv2d(64, 64, kernel_size=1, padding=0))


            print('================ variation added ===============' )

        
    def forward(self, inputVariable, mmaps_label, test=False): # label mmaps
    
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        
        if self.motion_segmentation:
            if self.classification:
                loss_ms = nn.CrossEntropyLoss()
            else:
                loss_ms = nn.L1Loss() 
        
        sum_loss_mmap = 0

        for t in range(inputVariable.size(0)): # iterate on the frames, default 7
            
            if test:
                b_list = []
                g_list = []
                f_conv = []
                f_var = []
            
            if self.variation:
                if t == 0: #first iteration we do not have the previous frame =>  skipped 
                    outB = 1
                    outY = 0
                    
                elif t > 0: # now we can exploit the output of the previous frame (out_conv100)
                    outT = self.transpose(out_conv_ms.detach())#detach otherwise you try to store all the computational graph
                    outB = self.betaConv(outT)
                    outY = self.gammaConv(outT)
                
                # apply the elemnt wise multiplication (beta) and the shift (gamma) to the selected layer in the resnet
                if test and not isinstance(outB, int):
                    logit, feature_conv, feature_convNBN, beta_np, gamma_np, f_conv1_np, f_variation_np = self.resNet(inputVariable[t], outB, outY, test=test)
                    b_list.append(beta_np)
                    g_list.append(gamma_np)
                    f_conv.append(f_conv1_np)
                    f_var.append(f_variation_np)
                else:
                    logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t], outB, outY)
            else:
                logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t], 1, 0)
            
            #spatial attention 
            if self.spatial_attention:
                bz, nc, h, w = feature_conv.size()
                feature_conv1 = feature_conv.view(bz, nc, h*w)
                probs, idxs = logit.sort(1, True)
                class_idx = idxs[:, 0]
                cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
                attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
                state = self.lstm_cell(attentionFeat, state)
                #print('===========spatial_attention==============')
            else:
                state = self.lstm_cell(feature_conv, state)
            
            if self.motion_segmentation: # motion segmentation part (with batch normalization!)
                
                #label motion segmentantion
                l_mmap = mmaps_label[t]
                
                out=self.relu_ms(feature_conv)
                out_conv_ms=self.conv100(out)
                out=torch.flatten(out_conv_ms, start_dim=1)
                out=self.fc1(out)
                
                if self.classification:
                    out = out.contiguous().view(out.size(0), 2, 7, 7) #batch,2,7,7
                else:
                    out = out.contiguous().view(out.size(0), 7, 7) #batch,7,7
                
                sum_loss_mmap += loss_ms(out, l_mmap)
           

        feats_avgpool = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats_classifier = self.classifier(feats_avgpool)
        
        if test:
            return feats_classifier, feats_avgpool, sum_loss_mmap, b_list, g_list, f_conv, f_var
        
        else:
            return feats_classifier, feats_avgpool, sum_loss_mmap