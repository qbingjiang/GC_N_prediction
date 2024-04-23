from torch import nn
import os
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import SimpleITK as sitk 
import skimage 

cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')
sam = sam_model_registry['vit_h'](checkpoint="/sam_checkpoint/sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

def extract_sam_feats(x_1_patch, y_1_patch): 
    # img = np.random.rand(96, 96, 96)
    # img*=255
    img = x_1_patch * y_1_patch
    img = img.astype(np.uint8) 
    feats_slice_pool_list = []
    feats_slice_permute_list = []
    feats_slice_ori_list = [] 
    feats_slice_list = []
    for i in range(len(img)): 
        img_slice = img[i,:,:,None]
        input_ = np.concatenate([img_slice, img_slice, img_slice],axis=2) 
        predictor.set_image(input_)
        feats_slice = predictor.features 

        y_1_patch_i_resize = skimage.transform.resize(y_1_patch[i], [64, 64], order=0, preserve_range=True, anti_aliasing=False ) 
        y_1_patch_i_resize_ = np.stack([y_1_patch_i_resize for _ in range(256)], axis=0)
        y_1_patch_i_resize_ = y_1_patch_i_resize_[None,:,:,:]
        y_1_patch_i_resize = torch.from_numpy(y_1_patch_i_resize_).float().to(device)
        feats_slice = feats_slice * y_1_patch_i_resize

        feats_slice_np = feats_slice.detach().to(torch.float16).cpu().numpy() 
        feats_slice_ori_list.append(feats_slice_np)
    feats_slice_ori_list_np = np.stack(feats_slice_ori_list, axis=0)
    return feats_slice_ori_list_np

def read_itk_files(img_path, label_path): 
    image_sitk = sitk.ReadImage( img_path ) 
    x = sitk.GetArrayFromImage(image_sitk) 
    originalimg_spacing = image_sitk.GetSpacing()
    label_sitk = sitk.ReadImage( label_path ) 
    y = sitk.GetArrayFromImage(label_sitk) 
    return x, y, originalimg_spacing

windowCenterWidth=(40, 400)
imgMinMax = [ windowCenterWidth[0] - windowCenterWidth[1]/2.0, windowCenterWidth[0] + windowCenterWidth[1]/2.0 ]
image_mask_files = [['pathtodata_image.nii.gz', 'pathtodata_mask.nii.gz']] 
for i in range(len(image_mask_files)):
    x_1_patch_, y_1_patch_, _ = read_itk_files(image_mask_files[i][0], image_mask_files[i][1] ) 
    x_1_patch_ = np.clip(x_1_patch_, a_min=imgMinMax[0], a_max=imgMinMax[1] ) 
    x_1_patch_ = (x_1_patch_ - imgMinMax[0] ) / (imgMinMax[1] - imgMinMax[0]) * 250 
    x_1_patch = x_1_patch_
    y_1_patch = (y_1_patch_ >0.5)*1 
    feats_slice_pool = extract_sam_feats( x_1_patch , y_1_patch ) 
    outfile_list = image_mask_files[i][0].replace('.nii.gz', '.npy').split('/')
    outfile = os.path.join(*outfile_list) 
    if not os.path.exists(outfile.rsplit('/',1)[0] ): os.makedirs(outfile.rsplit('/',1)[0] ) 
    np.save(outfile, feats_slice_pool) 
    print('save transformer features: ', outfile)

class Net_full(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv3d(96, 48, kernel_size=[8, 8, 8], padding=0, stride=[2,2,2])
        self.conv1 = nn.Conv3d(48, 24, 3, padding=0, stride=[2,2,2])
        self.conv2 = nn.Conv3d(24, 12, 3, padding=0, stride=[2,2,2])
        self.conv3 = nn.Conv3d(12, 6, 3, padding=0, stride=[2,2,2])
        self.fc = nn.Linear(6*14*2*2, 1)
        
        # Using a dictionary to store activations and gradients for multiple layers
        self.features = {}
        self.gradients = {}

    def register_hooks(self):
        layers = [self.conv0, self.conv1, self.conv2, self.conv3]
        for idx, layer in enumerate(layers):
            self.features[idx] = None  # Initialize storage for layer's activations
            self.gradients[idx] = None  # Initialize storage for layer's gradients
            
            def get_activations_hook(layer_idx):
                def hook(model, input, output):
                    self.features[layer_idx] = output.detach()
                return hook
            
            def get_gradients_hook(layer_idx):
                def hook(model, grad_input, grad_output):
                    self.gradients[layer_idx] = grad_output[0].detach()
                return hook
            
            layer.register_forward_hook(get_activations_hook(idx))
            # Use register_full_backward_hook for backward hooks, correct the signature
            layer.register_full_backward_hook(get_gradients_hook(idx))

    def get_activation_gradients(self, layer_idx):
        return self.gradients[layer_idx]

    def get_activations(self, layer_idx):
        return self.features[layer_idx] 
    
    def forward(self, x): 
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

def initialize_weights(model, init_method='xavier'):
    # He et. al weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            # Apply weight initialization to Conv2d and Linear layers
            if init_method=='xavier': 
                nn.init.xavier_uniform_(m.weight)  ##xavier_normal_
            if init_method=='orthogonal': 
                torch.nn.init.orthogonal(m.weight)
            if init_method=='kaiming': 
                torch.nn.init.kaiming_normal_(m.weight)  ##kaiming_uniform_
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d): 
            m.weight.data.fill_(1) 
            m.bias.data.zero_() 

model = Net_full()
initialize_weights(model)


print()

