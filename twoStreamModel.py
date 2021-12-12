import torch
from flow_resnet import *
from self_sup_model_convLSTM import *
import torch.nn as nn


class twoStreamAttentionModel(nn.Module):
    def __init__(self, flowModel='', frameModel='', stackSize=5, memSize=512, num_classes=61):
        super(twoStreamAttentionModel, self).__init__()
        
        self.flowModel = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
        
        if flowModel != '':
            self.flowModel.load_state_dict(torch.load(flowModel))
        
        self.frameModel = selfSupModel(num_classes, memSize, (True, False, False, False)) # #spatial_attention, motion_segmentation, variation, classification
        
        if frameModel != '':
            self.frameModel.load_state_dict(torch.load(frameModel))
        
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512 * 2, num_classes, bias=True))

    def forward(self, inputVariableFlow, inputVariableFrame):
        _, flowFeats, _ = self.flowModel(inputVariableFlow)
        _, rgbFeats, _= self.frameModel(inputVariableFrame, None)
        twoStreamFeats = torch.cat((flowFeats, rgbFeats), 1)
        return self.classifier(twoStreamFeats)