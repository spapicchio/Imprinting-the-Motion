from __future__ import print_function, division
from self_sup_model_convLSTM import *
from twoStreamModel import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
import make_dataset_self_supervised as datasetSupervised
import makeDatasetRGB as datasetRGB
import makeDatasetFlow as datasetFlow
import makeDatasetTwoStream as datasetTwoStream
from flow_resnet import *
import argparse
import sys
import os
from torchsummary import summary


def main_run(dataset, stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, memSize, flag_class, alpha, rgbModel, flowModel ):

    #============= possible flag ===============
    #default = resNet + attention + convLSTM
    #no_attention = resNet + convLSTM
    #flow = flow_resnet 
    #jointStream = flow_resnet | resnet + attention + convLSTM + fully connected
    #ms_class = add the motion segmentantion part treated as classification
    #ms_regr = add the motion segmentantion part treated as regression
    #variation_cl = ms_class + variation
    #variation_re = ms_regr + variation
    #=================================================
    print()
    if flag_class == 'default':
        print("============== default ===============")
        architecture_flag = (True, False, False, False) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetRGB.makeDataset
    
    if flag_class == 'no_attention':
        print("============== no_attention ===============")
        architecture_flag = (False, False, False, False) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetRGB.makeDataset
        
    if flag_class == 'flow':
        print("============== flow ===============")
        architecture_flag = (False, False, False, False) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetFlow.makeDataset
        
    if flag_class == 'jointStream':
        print("============== jointStream ===============")
        architecture_flag = (False, False, False, False) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetTwoStream.makeDataset
    
    if flag_class == 'ms_class':
        print("============== ms_class ===============")
        architecture_flag = (True, True, False, True) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetSupervised.makeDataset
    
    if flag_class == 'ms_regr':
        print("============== ms_regr ===============")
        architecture_flag = (True, True, False, False) #spatial_attention, motion_segmentation, variation, classification 
        makeDataset = datasetSupervised.makeDataset
        
    if flag_class == 'variation_cl':
        print("============== variation_cl ===============")
        architecture_flag = (True, True, True, True) #spatial_attention, motion_segmentation, variation, classification
        makeDataset = datasetSupervised.makeDataset
        
    if flag_class == 'variation_re':
        print("============== variation_re ===============")
        architecture_flag = (True, True, True, False) #spatial_attention, motion_segmentation, variation, classification 
        makeDataset = datasetSupervised.makeDataset
    print()
    if dataset.lower() == 'gtea61':
        num_classes = 61
    else:
        print('Dataset not found')
        sys.exit()

    model_folder = os.path.join('./', out_dir, dataset)  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)
    
    #=========== log file ============
    writer = SummaryWriter(model_folder)

    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    train_log_loss_acc = open((model_folder + '/train_loss_acc.txt'), 'w')

    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')
    val_log_loss_acc = open((model_folder + '/val_log_loss_acc.txt'), 'w')

    #=========== Data loader =================
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform_rgb_1 = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224)])
    spatial_transform_rgb_2 = Compose([ToTensor(), normalize])
    spatial_transform_mmaps = Compose([Scale(7), ToTensor()])
    
    vid_seq_train = makeDataset(train_data_dir,
                                spatial_transform=(spatial_transform_rgb_1,spatial_transform_rgb_2, spatial_transform_mmaps),
                                seqLen=seqLen,
                                fmt='.png',
                                flag_class = architecture_flag[3])

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)
                            
    if val_data_dir is not None:
        vid_seq_val = makeDataset(val_data_dir,
                                  spatial_transform=(Compose([Scale(256), CenterCrop(224)]), spatial_transform_rgb_2, spatial_transform_mmaps),
                                  seqLen=seqLen,
                                  train=False,
                                  fmt='.png',
                                  flag_class = architecture_flag[3],
                                  phase='val')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)

        valInstances = vid_seq_val.__len__()


    trainInstances = vid_seq_train.__len__()

    train_params = []

    if flag_class == 'flow':
        model = flow_resnet34(True, channels=2*seqLen, num_classes=num_classes)
        model.train(True)
        train_params = list(model.parameters())
    
    elif flag_class == 'jointStream':
        model = twoStreamAttentionModel(flowModel=flowModel, frameModel=rgbModel, stackSize=5, memSize=memSize, num_classes=num_classes)
        
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
        
        train_params = []
        
        model.classifier.train(True)
        for params in model.classifier.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.frameModel.lstm_cell.parameters():
            train_params += [params]
            params.requires_grad = True

        for params in model.frameModel.resNet.layer4.parameters():
            params.requires_grad = True
            train_params += [params]
        
        for params in model.frameModel.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]

        base_params = []
        for params in model.flowModel.layer4.parameters():
            base_params += [params]
            params.requires_grad = True
    
    elif stage == 1:
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
        
        #train lstm_cell
        model.lstm_cell.train(True)
        for params in model.lstm_cell.parameters():
            params.requires_grad = True
            train_params += [params]
        
        # train the classifier of the architecture
        model.classifier.train(True)
        for params in model.classifier.parameters():
            params.requires_grad = True
            train_params += [params]
       
    elif stage == 2:
        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
        model.load_state_dict(torch.load(stage1_dict), strict = False)
        
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
        
        #train only the layer 4 and the fc of the resnet
        model.resNet.layer4.train(True)
        for params in model.resNet.layer4.parameters():
            params.requires_grad = True
            train_params += [params]
            
        model.resNet.fc.train(True)
        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]
        
        #train lstm_cell
        model.lstm_cell.train(True)
        for params in model.lstm_cell.parameters():
            params.requires_grad = True
            train_params += [params]
        
        # train the classifier of the architecture
        model.classifier.train(True)
        for params in model.classifier.parameters():
            params.requires_grad = True
            train_params += [params]    
        
        if architecture_flag[1]: # motion_segmentation == True
           
            #train the motion segmentation part
            model.conv100.train(True)
            for params in model.conv100.parameters():
                params.requires_grad = True
                train_params += [params]
            
            model.fc1.train(True) 
            for params in model.fc1.parameters():
                params.requires_grad = True
                train_params += [params]
            
            if  architecture_flag[2]: #variation == True
                
                model.betaConv.train(True)
                for params in model.betaConv.parameters():
                    params.requires_grad = True
                    train_params += [params]   
                
                model.gammaConv.train(True)
                for params in model.gammaConv.parameters():
                    params.requires_grad = True
                    train_params += [params]

                model.resNet.conv1.train(True)
                for params in model.resNet.conv1.parameters():
                    params.requires_grad = True
                    train_params += [params]                    

    print()
    
    print(model) # ci da i layer del modello 
    
    for name, param in model.named_parameters():
        if param.requires_grad: 
            print (name)
    
    print()
    
    model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    
    
    if flag_class == 'jointStream':
        optimizer_fn = torch.optim.SGD([{'params': train_params},
                                        {'params': base_params, 'lr': 1e-4}],
                                        lr=lr1, momentum=0.9, weight_decay=5e-4)
    elif flag_class == 'flow':
        print("optimizer")
        optimizer_fn = torch.optim.SGD(train_params, lr=lr1, momentum=0.9, weight_decay=5e-4)
    
    else: 
        optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)
    

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step, gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0

    for epoch in range(numEpochs):
        if flag_class == 'flow':
          model.train(True)
    
        elif flag_class == 'jointStream':
          model.classifier.train(True)
          model.flowModel.layer4.train(True)

        elif stage == 1:
          model.lstm_cell.train(True)
          model.classifier.train(True)
          
        elif stage == 2:
          model.resNet.layer4.train(True)    
          model.resNet.fc.train(True)
          model.lstm_cell.train(True)
          model.classifier.train(True)
          
          if architecture_flag[1]: # motion_segmentation == True
              model.conv100.train(True)
              model.fc1.train(True) 
              
              if  architecture_flag[2]: #variation == True
                  model.betaConv.train(True)
                  model.gammaConv.train(True)
                  model.resNet.conv1.train(True)
    
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        
        
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        #architecture_flag => spatial_attention, motion_segmentation, variation, classification
        for i, (inputs, mmap_labels, targets) in enumerate(train_loader):
            #print(f"============= iteration {i} ===============")
            train_iter += 1
            iterPerEpoch += 1
            trainSamples += inputs.size(0)
            
            optimizer_fn.zero_grad()
            if flag_class == 'jointStream':
                inputVariableFlow = Variable(inputs.cuda())
                inputVariableFrame = Variable(mmap_labels.permute(1, 0, 2, 3, 4).cuda())
                labelVariable = Variable(targets.cuda())
                output_label = model(inputVariableFlow, inputVariableFrame)
                loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
                loss.backward()
                optimizer_fn.step()
                _, predicted = torch.max(output_label.data, 1)
                numCorrTrain += (predicted == targets.cuda()).sum()
                epoch_loss += loss.data

            else:
                
                if flag_class == 'flow':
                    inputVariable = Variable(inputs.cuda())
                else:
                    inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
                
                labelVariable = Variable(targets.cuda())
                
                if architecture_flag[1] or architecture_flag[2]: #only if we are calculating the motion segmentation
                    labelVariableMmap = Variable(mmap_labels.permute(1,0,2,3).cuda())
                    #print('========== label variable calculated========')
                    #print(f'mmap_labels:{mmap_labels.permute(1,0,2,3).size()}')
                    
                if architecture_flag[1] or architecture_flag[2]: #only for motion segmentation and variation
                    lstm_output, _, loss_mmap = model(inputVariable, labelVariableMmap)
                    #print('========== loss mmap calculated========')
                else:
                    lstm_output, _, _ = model(inputVariable, None)
                    loss_mmap = 0

                loss_lstm = loss_fn(lstm_output, labelVariable)

                loss_total = loss_lstm + alpha * loss_mmap 
                
                loss_total.backward()
                
                optimizer_fn.step()
                
                _, predicted = torch.max(lstm_output.data, 1)
                numCorrTrain += (predicted == targets.cuda()).sum()
                epoch_loss += loss_total.data
            
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy))
        

        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        
        train_log_loss.write(f'{avg_loss}\n')
        train_log_acc.write(f'{trainAccuracy}\n')
        train_log_loss_acc.write(f'epoch: {epoch+1} |loss: {avg_loss} |acc: {trainAccuracy}\n')
        
        
        if val_data_dir is not None:
            with torch.no_grad(): # Inserted in order to not calculate the gradient
                model.train(False)
                
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                
                for j, (inputs, mmap_labels, targets) in enumerate(val_loader):
                
                    val_iter += 1
                    val_samples += inputs.size(0)
                    
                    if flag_class == 'jointStream':
                        inputVariableFlow = Variable(inputs.cuda())
                        inputVariableFrame = Variable(mmap_labels.permute(1, 0, 2, 3, 4).cuda())
                        labelVariable = Variable(targets.cuda())
                        output_label = model(inputVariableFlow, inputVariableFrame)
                        loss = loss_fn(F.log_softmax(output_label, dim=1), labelVariable)
                        val_loss_epoch += loss.data
                        _, predicted = torch.max(output_label.data, 1)
                        numCorr += (predicted == labelVariable.data).sum()
                    else:
                        if flag_class == 'flow':
                            inputVariable = Variable(inputs.cuda())
                        else:
                            inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
                        labelVariable = Variable(targets.cuda(non_blocking=True)) 
                            
                        if architecture_flag[1]: #only if we are calculating the motion segmentation
                            labelVariableMmap = Variable(mmap_labels.permute(1, 0, 2, 3).cuda())
                            lstm_output, _, loss_mmap_val = model(inputVariable, labelVariableMmap)
                        else:
                            lstm_output, _, _ = model(inputVariable, None)
                            loss_mmap_val = 0

                        loss_lstm_val = loss_fn(lstm_output, labelVariable)                    
                        
                        val_loss = loss_lstm_val + alpha * loss_mmap_val
                        val_loss_epoch += val_loss.detach()

                        _, predicted = torch.max(lstm_output.detach(), 1)
                        numCorr += (predicted == targets.cuda()).sum()
                    
                val_accuracy = (numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                print()
                
                
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write(f'{avg_val_loss}\n')
                val_log_acc.write(f'{val_accuracy}\n')
                val_log_loss_acc.write(f'epoch: {epoch+1} |loss: {avg_val_loss} |acc: {val_accuracy}\n')
                    
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
       
        optim_scheduler.step()

    train_log_loss.close()
    train_log_acc.close()
    train_log_loss_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--stage', type=int, help='Training stage')
    parser.add_argument('--trainDatasetDir', type=str, help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, help='Val set directory')
    parser.add_argument('--outDir', type=str, help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='None', help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=7, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--alpha', type=float, default=1, help='loss1 + alpha * loss2')
    parser.add_argument('--flag', type=str, default ='default', help ='')
    parser.add_argument('--rgbModel', type=str, default ='rgbModel', help ='rgbModel dict for two joint stream')
    parser.add_argument('--flowModel', type=str, default ='flowModel', help ='flowModel dict for two joint stream')

    args = parser.parse_args()
    dataset = args.dataset
    stage = args.stage
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize
    alpha = args.alpha
    flag = args.flag
    rgbModel = args.rgbModel
    flowModel = args.flowModel
  
    main_run(dataset, stage, trainDatasetDir, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize, flag, alpha, rgbModel, flowModel)

__main__()