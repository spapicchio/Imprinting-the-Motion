from __future__ import print_function, division
from self_sup_model_convLSTM import *
from flow_resnet import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import sys
import make_dataset_self_supervised as datasetSupervised
import makeDatasetRGB as datasetRGB
import makeDatasetFlow as datasetFlow
import makeDatasetTwoStream as datasetTwoStream
import pandas as pd
import matplotlib.pyplot as plt
import os
def main_run(dataset, model_state_dict, dataset_dir, seqLen, memSize, flag_class):
    
    if dataset.lower() == 'gtea61':
        num_classes = 61
    else:
        print('Dataset not found')
        sys.exit()

    if flag_class == 'default':
        architecture_flag = (True, False, False, False) #spatial_attention, motion_segmentation, variation, classification   makeDataset = datasetRGB.makeDataset
        makeDataset = datasetRGB.makeDataset
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
    
    if flag_class == 'no_attention':
        architecture_flag = (False, False, False, False) #spatial_attention, motion_segmentation, variation, classification   makeDataset = datasetRGB.makeDataset
        makeDataset = datasetRGB.makeDataset
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
        
    if flag_class == 'flow':
        makeDataset = datasetFlow.makeDataset
        architecture_flag = (False, False, False, False)
        model = flow_resnet34(False, channels=2*seqLen, num_classes=num_classes)
        
    if flag_class == 'jointStream':
        makeDataset = datasetTwoStream.makeDataset
        architecture_flag = (False, False, False, False)
        model = twoStreamAttentionModel(stackSize=5, memSize=512, num_classes=num_classes)
        

    
    if flag_class == 'ms_class':
        architecture_flag = (True, True, False, True) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetSupervised.makeDataset
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
    
    if flag_class == 'ms_regr':
        architecture_flag = (True, True, False, False) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetSupervised.makeDataset
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
        
    if flag_class == 'variation_cl':
        architecture_flag = (True, True, True, True) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetSupervised.makeDataset
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
        
    if flag_class == 'variation_re':
        architecture_flag = (True, True, True, False) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetSupervised.makeDataset
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
    
    #load the weight
    model.load_state_dict(torch.load(model_state_dict))

    
        
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)
    spatial_transform_1 = Compose([Scale(256), CenterCrop(224)])
    spatial_transform_2 = Compose([ToTensor(), normalize])
    spatial_transform_mmap = Compose([Scale(7), ToTensor()])

    vid_seq_test = makeDataset(dataset_dir,
                               spatial_transform=(spatial_transform_1, spatial_transform_2, spatial_transform_mmap),
                               train = False,
                               seqLen=seqLen, fmt='.png',
                               flag_class = architecture_flag[3])

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=1,
                            shuffle=False, num_workers=2, pin_memory=True)


    with torch.no_grad():
        for params in model.parameters():
            params.requires_grad = False

        model.train(False)
        model.cuda()

        test_samples = vid_seq_test.__len__()
        print(f'Number of samples = {test_samples}')
        print('Evaluating...')
        
        numCorr = 0
        true_labels = []
        predicted_labels = []

        for j, (inputs, mmap_labels, target) in enumerate(test_loader):
            
            if flag_class == 'flow':
                inputVariable = Variable(inputs[0].cuda())
                output_label, _ = model(inputVariable)
                output_label_mean = torch.mean(output_label.data, 0, True)
                _, predicted = torch.max(output_label_mean, 1)
                numCorr += (predicted == target).sum()
                true_labels.append(target)
                predicted_labels.append(predicted)
            
            elif flag_class == 'jointStream':
                inputVariableFlow = Variable(inputs.cuda())
                inputVariableFrame = Variable(mmap_labels.permute(1, 0, 2, 3, 4).cuda())
                output_label = model(inputVariableFlow, inputVariableFrame)
                _, predictedTwoStream = torch.max(output_label.detach(), 1)
                numCorr += (predictedTwoStream == target.cuda()).sum()
                predicted_labels.append(predictedTwoStream)
                true_labels.append(targets)
                        
            elif flag_class == 'no_attention' or flag_class == 'default':
                inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda(), volatile=True)
                output_label, _, _ = model(inputVariable, None)
                _, predicted = torch.max(output_label.detach(), 1)
                numCorr += (predicted == target.cuda()).sum()
                true_labels.append(target)
                predicted_labels.append(predicted)
                
            else: 
                inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
                labelVariableMmap = Variable(mmap_labels.permute(1,0,2,3).cuda())
                test=False

                if test:
                    output_label, _, _, b_list, g_list, f_conv, f_var = model(inputVariable,labelVariableMmap, test=test)
                else:
                    output_label, _, _ = model(inputVariable,labelVariableMmap)
                    
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == target.cuda()).sum()
                true_labels.append(target)
                
                predicted_labels.append(predicted)
                
                if test:
                    my_dir = '/content/images_features/'
                    print(int(target[0]))
                    my_dir_label = os.path.join(my_dir, f'id_{int(target[0])}')
                    if not os.path.exists(my_dir):
                        os.mkdir(my_dir)
                    if not os.path.exists(my_dir_label):
                        os.mkdir(my_dir_label)
                    for i in range(b_list[0].shape[0]):
                        if not os.path.exists(os.path.join(my_dir_label, f'channel_{i}')):
                            os.mkdir(os.path.join(my_dir_label, f'channel_{i}'))
                        print(b_list[0].shape[0])
                        print(b_list[0][10].shape)
                        plt.imsave(os.path.join(my_dir_label, f'channel_{i}', 'beta.png'), b_list[0][i,...], cmap='Greys')
                        plt.imsave(os.path.join(my_dir_label, f'channel_{i}', 'gamma.png'), g_list[0][i,...], cmap='Greys')
                        plt.imsave(os.path.join(my_dir_label, f'channel_{i}', 'conv.png'), f_conv[0][i,...], cmap='Greys')
                        plt.imsave(os.path.join(my_dir_label, f'channel_{i}', 'var.png'), f_var[0][i,...], cmap='Greys')
                        print(f"===== saved in{os.path.join(my_dir_label, f'channel{i}')}")
                        

        test_accuracy = (numCorr / test_samples) * 100
        print(f'Test Accuracy = {test_accuracy}')
        
        id_label = dict()
        root_dir = os.path.join(dataset_dir, 'processed_frames2')
        for dir_user in sorted(os.listdir(root_dir)): 
          
            
          if dir_user!='S2':
            continue
          
    
          class_id = 0
          if not os.path.isdir(os.path.join(root_dir, dir_user)):
              continue
            
          dir = os.path.join(root_dir, dir_user)
          for target in sorted(os.listdir(dir)):
              
              id_label[class_id] = target
              class_id += 1
        
        #confusion_matrix = dict() # key= True_label, value=[truePrediction, falsePrediction]
        
        true_labes_mapped = [id_label[int(t[0])] for t in true_labels]
        predicted_labels_mapped = [id_label[int(t[0])] for t in predicted_labels]
        label=sorted( list(id_label.values()) )
        cm = confusion_matrix(true_labes_mapped, predicted_labels_mapped, labels=label )
        pd.set_option('display.max_colwidth', None)
        confusion=pd.DataFrame(cm, columns=label, index=label)
        confusion.to_csv("/content/confusion.csv")
        print('confusion matrix created in /content/confusion.csv')

        d_true_pred = {}
        for t,p in zip(true_labes_mapped,predicted_labels_mapped):
            if t in d_true_pred:
                v=d_true_pred[t]
                v.append(p)
                d_true_pred[t]=v
            else:
                d_true_pred[t]=[p]
        print(d_true_pred)
        
        dict_log = open(('/content/true_pred_log.txt'), 'w')
        for k in d_true_pred:
            dict_log.write(f'key: {k}, value: {d_true_pred[k]}\n')
        dict_log.close()

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default='no default',
                        help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str, default='no default',
                        help='Model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--flag', type=str, help='classification or regression?')

    args = parser.parse_args()

    dataset = args.dataset
    model_state_dict = args.modelStateDict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    memSize = args.memSize
    flag = args.flag

    main_run(dataset, model_state_dict, dataset_dir, seqLen, memSize, flag)

__main__()