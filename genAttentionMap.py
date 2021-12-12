import numpy as np
from torchvision import transforms
import cv2
from self_sup_model_convLSTM import *
from attentionMapModel import attentionMap
from PIL import Image
import argparse
import os
import glob

####################Model definition###############################
#num_classes = 61 # Classes in the pre-trained model
#mem_size = 512
#model_state_dict = '/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/ConvLSTM_Attention/output_stage_2_16/gtea61/rgb/stage2/model_rgb_state_dict.pth' # Weights of the pre-trained model

#model = attentionModel(num_classes=num_classes, mem_size=mem_size)
#model.load_state_dict(torch.load(model_state_dict))
#model_backbone = model.resNet
#attentionMapModel = attentionMap(model_backbone).cuda()
#attentionMapModel.train(False)
#for params in attentionMapModel.parameters():
    #params.requires_grad = False
###################################################################

def main_run(dir_in, dir_out, model_state_dict, mem_size, num_classes, flag_class):

    if flag_class == 'default':
        architecture_flag = (True, False, False, False) #spatial_attention, motion_segmentation, variation, classification   makeDataset = datasetRGB.makeDataset

        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
    
    if flag_class == 'no_attention':
        architecture_flag = (False, False, False, False) #spatial_attention, motion_segmentation, variation, classification   makeDataset = datasetRGB.makeDataset

        model = selfSupModel(num_classes=num_classes, mem_size=memSize, list_flags=architecture_flag)
        
    if flag_class == 'flow':

        architecture_flag = (False, False, False, False)
        model = flow_resnet34(False, channels=2*seqLen, num_classes=num_classes)
        
    if flag_class == 'jointStream':

        architecture_flag = (False, False, False, False)
        model = twoStreamAttentionModel(stackSize=5, memSize=512, num_classes=num_classes)
    
    if flag_class == 'ms_class':
        architecture_flag = (True, True, False, True) #spatial_attention, motion_segmentation, variation, classification

        model = selfSupModel(num_classes=num_classes, mem_size=512, list_flags=architecture_flag)
    
    if flag_class == 'ms_regr':
        architecture_flag = (True, True, False, False) #spatial_attention, motion_segmentation, variation, classification

        model = selfSupModel(num_classes=num_classes, mem_size=512, list_flags=architecture_flag)
        
    if flag_class == 'variation_cl':
        architecture_flag = (True, True, True, True) #spatial_attention, motion_segmentation, variation, classification

        model = selfSupModel(num_classes=num_classes, mem_size=512, list_flags=architecture_flag)
        
    if flag_class == 'variation_re':
        architecture_flag = (True, True, True, False) #spatial_attention, motion_segmentation, variation, classification
        model = selfSupModel(num_classes=num_classes, mem_size=512, list_flags=architecture_flag)
    
    #load the weight
    model.load_state_dict(torch.load(model_state_dict))

    model_backbone = model.resNet
    attentionMapModel = attentionMap(model_backbone).cuda()
    attentionMapModel.train(False)
    
    for params in attentionMapModel.parameters():
        params.requires_grad = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    
    preprocess1 = transforms.Compose([transforms.Scale(256),
                                     transforms.CenterCrop(224)])

    preprocess2 = transforms.Compose([transforms.ToTensor(),
                                      normalize])

    os.mkdir(dir_out)

    for target in sorted(os.listdir(dir_in)):
        
        if not os.path.isdir(os.path.join(dir_in, target)): #to skip the .DS Store file
            continue

        dir1 = os.path.join(dir_in, target)
        dir_t_out=  os.path.join(dir_out, target)
        os.mkdir(dir_t_out)
        insts = sorted(os.listdir(dir1))

        if insts != []:
            
            for inst in insts:
                dir_inst_out=os.path.join(dir_t_out, inst)
                os.mkdir(dir_inst_out)
                inst_dir = os.path.join(dir1, inst, 'rgb') #directly goes to rgb folder
                inst_dir_out = os.path.join(dir_inst_out,'map')
                os.mkdir(inst_dir_out)

                gif_immages = []
                for f in sorted(os.listdir(inst_dir)):
                    img_pil = Image.open(os.path.join(inst_dir,f))
                    img_pil1 = preprocess1(img_pil)
                    img_size = img_pil1.size
                    size_upsample = (img_size[0], img_size[1])
                    img_tensor = preprocess2(img_pil1)
                    img_variable = Variable(img_tensor.unsqueeze(0).cuda())
                    img = np.asarray(img_pil1)
                    attentionMap_image = attentionMapModel(img_variable, img, size_upsample)
                   
                    cv2.imwrite(os.path.join(inst_dir_out,f), attentionMap_image)
                
                # gif creation
                fp_in = os.path.join(inst_dir_out,'*.png')
                fp_out = os.path.join(inst_dir_out, 'image.gif')

                img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
                img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=150, loop=0)
    
def __main__():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_in', type=str, default='/content/gtea61', help='input dir')
    parser.add_argument('--dir_out', type=str, default='content/output/', help='output dir')
    parser.add_argument('--model_dict', type=str, default='/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/ConvLSTM_Attention/output_stage_2_16/gtea61/rgb/stage2/model_rgb_state_dict.pth',
                        help="model's weights")
    parser.add_argument('--mem_size', type=int, default=512,
                        help='ConvLSTM hidden state size')
    parser.add_argument('--num_classes', type=int, default=61, help='')
    parser.add_argument('--flag', type=str, help='correct flag for the model')

    args = parser.parse_args()

    main_run(args.dir_in, args.dir_out, args.model_dict, args.mem_size, args.num_classes, args.flag)

__main__()




#dir_in = '/content/GTEA61/processed_frames2/S2/pour_water,cup/1/rgb'
#fl_name_out = '/content/drive/MyDrive/Machine_Learning_Project/Google_Colabs/ConvLSTM_Attention/MAP/Attention1'
#for f in sorted(os.listdir(dir_in)):
#  img_pil = Image.open(os.path.join(dir_in,f))
#  img_pil1 = preprocess1(img_pil)
#  img_size = img_pil1.size
#  size_upsample = (img_size[0], img_size[1])
 # img_tensor = preprocess2(img_pil1)
 # img_variable = Variable(img_tensor.unsqueeze(0).cuda())
 # img = np.asarray(img_pil1)
 # attentionMap_image = attentionMapModel(img_variable, img, size_upsample)
 # cv2.imwrite(os.path.join(fl_name_out,f), attentionMap_image)



