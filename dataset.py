import torch
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio
import SimpleITK as sitk
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import glob
import itertools
import random
from random import uniform
#from imgaug import augmenters as iaa
from skimage.transform import radon, resize
import cv2
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_erosion
import sys
# import h5py
from scipy.ndimage import rotate, interpolation, zoom
import platform
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data._utils.collate import default_collate
import pydicom
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes, binary_erosion, distance_transform_edt
from skimage.segmentation import slic
# from fast_slic import Slic
# from fast_slic.avx2 import SlicAvx2
from medpy.io import load
import imageio
import re
from skimage.measure import label   
from skimage.transform import PiecewiseAffineTransform, warp, warp_coords

def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def gamma_contrast(image):
    p05 = np.percentile(image, 0.5)
    p995 = np.percentile(image, 99.5)
    
    diff = p995-p05+1e-7
    
    image = diff*((image/diff)**uniform(0.5,2))
    
    
    
    return image

def get_random_crop(image_list, crop_height, crop_width):

    max_x = image_list[0].shape[1] - crop_width
    max_y = image_list[0].shape[0] - crop_height
    
    try:
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
    except:
        image_list = [np.pad(a, ((0, max(0,-max_y+1)), (0,  max(0,-max_x+1))), 'constant') for a in image_list]
#        print(image_list[0].shape)
        max_x = image_list[0].shape[1] - crop_width
        max_y = image_list[0].shape[0] - crop_height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

    crop = [image[y: y + crop_height, x: x + crop_width] for image in image_list]

    return crop

def cap_image(i, low, up):
    image = i.copy()
    image[image<low]=low
    image[image>up] = up
    
    image-=low
    image=image/(up-low)
    image*=255
    
    return image

def edge_profile(img_list, with_image=False, radius=3, interval=1):
    radius = radius
    interval = interval
    padding = radius*interval # multiply interval
    max_displacement = padding
    

    
    img_list_update = []
    for img in img_list:
        x_1 = img.transpose(1,2).transpose(2,3) #b h w c
        x_2 = F.pad(img, tuple([padding for _ in range(4)]), mode='replicate').transpose(1,2).transpose(2,3)
        
        # TODO need optimize
        out_vb = torch.zeros(1)
        _y=0
        _x=0
        for _y in range(0,max_displacement*2+1,interval):
            for _x in range(0,max_displacement*2+1,interval):
                if _y==max_displacement and _x==max_displacement:
                    continue
                c_out = (torch.sum(x_1-x_2[:, _x:_x+x_1.size(1), _y:_y+x_1.size(2), :],3, keepdim=True)).transpose(2,3).transpose(1,2) #b c h w
            
                out_vb = torch.cat((out_vb, c_out),1) if len(out_vb.size())!=1 else c_out
                
        out_vb = nn.Softmax(dim=1)(out_vb)
        if with_image:
            # x_1*=0
            out_vb = torch.cat([out_vb, x_1.transpose(2,3).transpose(1,2)], dim=1)
        img_list_update.append(out_vb)
        
    return img_list_update

class Dataset_all_abdomen(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_list, batch_size, set_size, mode='training'):
        'Initialization'
        self.mode = mode
        self.batch_size = batch_size
        self.set_size = set_size
        self.create_list_and_ID(image_list)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        try:
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            datatype = self.info[ID]['datatype']
            
            
            if datatype=='c4' or datatype=='ln' or datatype=='pa' or datatype=='co':
                f1 = self.info[ID]['f1']
                f2 = self.info[ID]['f2']
                
                # groundtruth
                ds = pydicom.read_file(f1)
                frame1 = ds.pixel_array.astype(float)*ds[0x0028, 0x1053].value+ds[0x0028, 0x1052].value
                ds = pydicom.read_file(f2)
                frame2 = ds.pixel_array.astype(float)*ds[0x0028, 0x1053].value+ds[0x0028, 0x1052].value
            
            'rotate'
            angle = random.randint(0, 359)
            frame1 = rotate(frame1, angle, reshape=False, mode='nearest')
            frame2 = rotate(frame2, angle, reshape=False, mode='nearest')
            
            resize_size = random.randint(256, 300)
            frame1 = resize(frame1, (resize_size,resize_size), anti_aliasing=True)
            frame2 = resize(frame2, (resize_size,resize_size), anti_aliasing=True)
            [frame1, frame2] = get_random_crop([frame1, frame2], 256, 256)  #224
            
            
            # agmentation for information bottleneck
            frame1_input = frame1.copy()
            frame2_input = frame2.copy()
            
            # frame1 = cap_image(frame1, -300, 400)    #-200, 300
            # frame2 = cap_image(frame2, -300, 400)    #-200, 300
            
            frame1_input = cap_image(frame1_input, random.randint(-400, -200), random.randint(300, 500))    #-200, 300
            frame2_input = cap_image(frame2_input, random.randint(-400, -200), random.randint(300, 500))    #-200, 300
            
            if np.std(frame1_input)<10 or np.std(frame2_input)<10:
                frame1_input+=(np.random.normal(loc=1, scale=uniform(0,0.4), size=frame1_input.shape))**2
                frame2_input+=(np.random.normal(loc=1, scale=uniform(0,0.4), size=frame1_input.shape))**2
                frame1+=(np.random.normal(loc=1, scale=uniform(0,0.4), size=frame1_input.shape))**2
                frame2+=(np.random.normal(loc=1, scale=uniform(0,0.4), size=frame1_input.shape))**2
            
            # frame1_input = frame1_input/255
            # frame2_input = frame2_input/255
            
            frame1_input = gamma_contrast(frame1_input)
            frame2_input = gamma_contrast(frame2_input)
            
            
    
            # frame1_input = frame1_input*np.random.normal(loc=1, scale=uniform(0,0.4), size=frame1_input.shape)
            # frame2_input = frame2_input*np.random.normal(loc=1, scale=uniform(0,0.4), size=frame2_input.shape)
            

            
            # make dimension right
            if len(frame1_input.shape)==2:
                frame1_input = frame1_input[np.newaxis,...]
                frame2_input = frame2_input[np.newaxis,...]
            if len(frame1.shape)==2:
                frame1 = frame1[np.newaxis,...]
                frame2 = frame2[np.newaxis,...]
                
                
            return frame1_input, frame2_input, frame1, frame2
        except Exception as e:
            a=1
        
    def create_list_and_ID(self, images):
        self.list_IDs = []
        self.info = {}
        self.vol_store={}
        
        
        
        count = 0
        for n, subfolder in enumerate(images):
            datatype = subfolder[0:2]
            data = subfolder[2:]
            
            if datatype=='c4' or datatype=='ln' or datatype=='pa':
                lstFilesDCM = glob.glob(os.path.join(data, '*dcm'))
            
                interval = np.random.choice([2,3,4],p=([1/3]*3))
                # each pair of frames
                for f in range(0,len(lstFilesDCM),interval+1):
                    temp = {}
                    if f+interval<len(lstFilesDCM):
                        if np.random.choice([1,2],p=[0.5,0.5])==1:
                            temp.update({'datatype':datatype, 'f1':lstFilesDCM[f], 'f2':lstFilesDCM[f+interval]})
                        else:
                            temp.update({'datatype':datatype, 'f2':lstFilesDCM[f], 'f1':lstFilesDCM[f+interval]})
                        self.info.update({count:temp})
                        self.list_IDs.append(count)
                        count+=1
            elif datatype=='co':
                lstFilesDCM = glob.glob(os.path.join(data, '*dcm'))
                if len(lstFilesDCM)<100:
                    continue
                interval = np.random.choice([2,3,4],p=([1/3]*3))
                # each pair of frames
                for f in range(0,len(lstFilesDCM),interval+1):
                    temp = {}
                    if f+interval<len(lstFilesDCM):
                        if np.random.choice([1,2],p=[0.5,0.5])==1:
                            temp.update({'datatype':datatype, 'f1':lstFilesDCM[f], 'f2':lstFilesDCM[f+interval]})
                        else:
                            temp.update({'datatype':datatype, 'f2':lstFilesDCM[f], 'f1':lstFilesDCM[f+interval]})
                        self.info.update({count:temp})
                        self.list_IDs.append(count)
                        count+=1
            
            else:
                print(subfolder)
          
        print('')
        self.shuffle_list()
        
        
    def shuffle_list(self):
        random.shuffle(self.list_IDs)
#        random.shuffle(self.sampling_params)

class Dataset_C4KC_KiTS(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_list, batch_size, set_size, mode='training'):
        'Initialization'
        self.mode = mode
        self.batch_size = batch_size
        self.set_size = set_size
        self.create_list_and_ID(image_list)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        try:
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            f1 = self.info[ID]['f1']
            f2 = self.info[ID]['f2']
            
            # groundtruth
            ds = pydicom.read_file(f1)
            frame1 = ds.pixel_array.astype(float)#*ds[0x0028, 0x1053].value+ds[0x0028, 0x1052].value
            ds = pydicom.read_file(f2)
            frame2 = ds.pixel_array.astype(float)#*ds[0x0028, 0x1053].value+ds[0x0028, 0x1052].value
            
            
            resize_size = random.randint(256, 300)
            frame1 = resize(frame1, (resize_size,resize_size), anti_aliasing=True)
            frame2 = resize(frame2, (resize_size,resize_size), anti_aliasing=True)
            [frame1, frame2] = get_random_crop([frame1, frame2], 256, 256)  #224
            
            
            # agmentation for information bottleneck
            frame1_input = frame1.copy()
            frame2_input = frame2.copy()
            
            frame1_input = cap_image(frame1_input, random.randint(-400, -200), random.randint(300, 500))    #-200, 300
            frame2_input = cap_image(frame2_input, random.randint(-400, -200), random.randint(300, 500))    #-200, 300
            
            # if np.std(frame1_input)<10 or np.std(frame2_input)<10:
            #     f=dsfl/2
            
            # frame1_input = frame1_input/255
            # frame2_input = frame2_input/255
            
            # frame1_input = gamma_contrast(frame1_input)
            # frame2_input = gamma_contrast(frame2_input)
            
            
    
            # frame1_input = frame1_input*np.random.normal(loc=1, scale=uniform(0,0.4), size=frame1_input.shape)
            # frame2_input = frame2_input*np.random.normal(loc=1, scale=uniform(0,0.4), size=frame2_input.shape)
            

            
            # make dimension right
            if len(frame1_input.shape)==2:
                frame1_input = frame1_input[np.newaxis,...]
                frame2_input = frame2_input[np.newaxis,...]
            if len(frame1.shape)==2:
                frame1 = frame1[np.newaxis,...]
                frame2 = frame2[np.newaxis,...]
                
            return frame1, frame2, frame1, frame2
        except Exception as e:
            print(e)
        
    def create_list_and_ID(self, images):
        self.list_IDs = []
        self.info = {}
        
        count = 0
        for n, subfolder in enumerate(images):
            if n%1000==0:
                print(n, end=' ') 
            lstFilesDCM = glob.glob(os.path.join(subfolder, '*dcm'))
            
            interval = np.random.choice([2,3,4],p=([1/3]*3))
            # each pair of frames
            for f in range(0,len(lstFilesDCM),interval+1):
                temp = {}
                if f+interval<len(lstFilesDCM):
                    if np.random.choice([1,2],p=[0.5,0.5])==1:
                        temp.update({'f1':lstFilesDCM[f], 'f2':lstFilesDCM[f+interval]})
                    else:
                        temp.update({'f2':lstFilesDCM[f], 'f1':lstFilesDCM[f+interval]})
                    self.info.update({count:temp})
                    self.list_IDs.append(count)
                    count+=1
       
           
        print('')
        self.shuffle_list()
        
        
    def shuffle_list(self):
        random.shuffle(self.list_IDs)
#        random.shuffle(self.sampling_params)

class Dataset_test_decathon_liver(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_list, batch_size, set_size, mode='training'):
        'Initialization'
        self.mode = mode
        self.batch_size = batch_size
        self.set_size = set_size
        self.create_list_and_ID(image_list)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        try:
            'Generates one sample of data'
            # Select sample
            file = self.list_IDs[index]
            
            img_vol = self.vol_store[file].astype(float)
            mask_vol = self.mask_store[file].astype(float)/255
            # mask_vol[mask_vol!=1] = 0
            mask_vol[mask_vol>0] = 1
            
            ###
            # img_vol = zoom(img_vol, (0.5,0.5,1))
            # mask_vol = zoom(mask_vol, (0.5,0.5,1), order=0)
        
            ###
            img_vol_norm = img_vol.copy()
            img_vol_norm = cap_image(img_vol_norm, -300, 400)    #-200, 300
            img_vol = img_vol_norm
            
            
            max_area=0
            for i in range(mask_vol.shape[-1]):
                if np.sum(mask_vol[:,:,i])>max_area:
                    max_area = np.sum(mask_vol[:,:,i])
                    pos_max=i
            
            pos = np.nonzero(np.sum(mask_vol, (0,1)))[0]
            
            # dimension
            img_vol_norm = np.moveaxis(img_vol_norm, [0, 1, 2], [1, 2, 0])
            img_vol = np.moveaxis(img_vol, [0, 1, 2], [1, 2, 0])
            mask_vol = np.moveaxis(mask_vol, [0, 1, 2], [1, 2, 0])
            
            return img_vol_norm, mask_vol, img_vol, pos_max, pos[0], pos[-1], file
        
        except Exception as e:
            print(e)
        
    def create_list_and_ID(self, images):
        self.list_IDs = []
        self.vol_store={}
        self.mask_store={}
        
        for n, file in enumerate(images):
            # print(n, end=' ')
            vol_file = file.replace('labelsTr', 'imagesTr')
            img_vol = nib.load(vol_file).get_fdata()
            mask_vol = nib.load(file).get_fdata()
        
            self.vol_store.update({file:img_vol})
            self.mask_store.update({file:mask_vol})
            self.list_IDs.append(file)
                
        # print('')
#        self.shuffle_list()
        
        
    def shuffle_list(self):
        random.shuffle(self.list_IDs)
#        random.shuffle(self.sampling_params)

# class Dataset_numpy_3d(data.Dataset):
#     'Characterizes a dataset for PyTorch loading 3D numpy volumes'
#     def __init__(self, folders_list, batch_size, set_size, mode='training'):
#         'Initialization'
#         self.mode = mode
#         self.batch_size = batch_size
#         self.set_size = set_size
#         self.create_list_and_ID(folders_list)

#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         try:
#             'Generates one sample of data'
#             # Select sample
#             ID = self.list_IDs[index]
#             f1_idx = self.info[ID]['f1']
#             f2_idx = self.info[ID]['f2']
#             vol_key = self.info[ID]['vol']
            
#             # Load volume dictionary and extract image
#             vol_dict = np.load(vol_key, allow_pickle=True).item()
#             vol = vol_dict['image'].astype(float)
            
#             # Extract slices
#             frame1 = vol[:, :, f1_idx]
#             frame2 = vol[:, :, f2_idx]
            
#             # Rotate
#             angle = random.randint(0, 359)
#             frame1 = rotate(frame1, angle, reshape=False, mode='nearest')
#             frame2 = rotate(frame2, angle, reshape=False, mode='nearest')
            
#             # Resize
#             resize_size = random.randint(256, 300)
#             frame1 = resize(frame1, (resize_size, resize_size), anti_aliasing=True)
#             frame2 = resize(frame2, (resize_size, resize_size), anti_aliasing=True)
#             [frame1, frame2] = get_random_crop([frame1, frame2], 256, 256)
            
#             # Augmentation for information bottleneck
#             frame1_input = frame1.copy()
#             frame2_input = frame2.copy()
            
#             frame1_input = cap_image(frame1_input, random.randint(-400, -200), random.randint(300, 500))
#             frame2_input = cap_image(frame2_input, random.randint(-400, -200), random.randint(300, 500))
            
#             if np.std(frame1_input) < 10 or np.std(frame2_input) < 10:
#                 frame1_input += (np.random.normal(loc=1, scale=uniform(0, 0.4), size=frame1_input.shape)) ** 2
#                 frame2_input += (np.random.normal(loc=1, scale=uniform(0, 0.4), size=frame2_input.shape)) ** 2
#                 frame1 += (np.random.normal(loc=1, scale=uniform(0, 0.4), size=frame1_input.shape)) ** 2
#                 frame2 += (np.random.normal(loc=1, scale=uniform(0, 0.4), size=frame2_input.shape)) ** 2
            
#             frame1_input = gamma_contrast(frame1_input)
#             frame2_input = gamma_contrast(frame2_input)
            
#             # Make dimension right
#             if len(frame1_input.shape) == 2:
#                 frame1_input = frame1_input[np.newaxis, ...]
#                 frame2_input = frame2_input[np.newaxis, ...]
#             if len(frame1.shape) == 2:
#                 frame1 = frame1[np.newaxis, ...]
#                 frame2 = frame2[np.newaxis, ...]
            
#             return frame1_input, frame2_input, frame1, frame2
        
#         except Exception as e:
#             print(f"Error in Dataset_numpy_3d.__getitem__: {e}")
#             return None
        
#     def create_list_and_ID(self, folders_list):
#         self.list_IDs = []
#         self.info = {}
        
#         count = 0
#         for n, folder in enumerate(folders_list):
#             print(f"Processing folder {n}: {folder}")
            
#             # Find all .npy files in folder
#             npy_files = sorted(glob.glob(os.path.join(folder, '*.npy')))
            
#             for npy_file in npy_files:
#                 try:
#                     # Load the 3D volume dictionary
#                     vol_dict = np.load(npy_file, allow_pickle=True).item()
#                     vol = vol_dict['image'].astype(float)
                    
#                     # Get number of slices in z dimension (last dimension)
#                     num_slices = vol.shape[-1]
                    
#                     if num_slices < 4:
#                         continue
                    
#                     # Create pairs of slices
#                     interval = np.random.choice([2, 3, 4], p=([1/3] * 3))
                    
#                     for f in range(0, num_slices, interval + 1):
#                         if f + interval < num_slices:
#                             temp = {}
#                             if np.random.choice([1, 2], p=[0.5, 0.5]) == 1:
#                                 temp.update({'vol': npy_file, 'f1': f, 'f2': f + interval})
#                             else:
#                                 temp.update({'vol': npy_file, 'f2': f, 'f1': f + interval})
#                             self.info.update({count: temp})
#                             self.list_IDs.append(count)
#                             count += 1
                
#                 except Exception as e:
#                     print(f"Error loading {npy_file}: {e}")
#                     continue
        
#         print(f"Total samples created: {len(self.list_IDs)}")
#         self.shuffle_list()
        
#     def shuffle_list(self):
#         random.shuffle(self.list_IDs)


class Dataset_numpy_3d(data.Dataset):
    """
    PyTorch dataset for 3D numpy volume dictionaries saved as .npy.

    Each .npy file is expected to be a dict with key 'image' -> 3D array [H, W, Z].
    The dataset builds a list of slice-pairs (f1, f2) from each volume.

    Input `sources` can be:
      - list of folders containing .npy files
      - list of .npy file paths
      - a mix of both
    """
    def __init__(self, sources, batch_size, set_size, mode='training',
                 slice_interval_choices=(2, 3, 4), seed=10):
        self.mode = mode
        self.batch_size = batch_size
        self.set_size = set_size
        self.slice_interval_choices = slice_interval_choices
        self.seed = seed

        self.list_IDs = []
        self.info = {}
        self._build_index(sources)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        try:
            ID = self.list_IDs[index]
            f1_idx = self.info[ID]['f1']
            f2_idx = self.info[ID]['f2']
            vol_path = self.info[ID]['vol']

            vol_dict = np.load(vol_path, allow_pickle=True).item()
            vol = vol_dict['image'].astype(np.float32)  # [H,W,Z]

            # basic sanity
            if vol.ndim != 3 or vol.shape[-1] <= max(f1_idx, f2_idx):
                return None

            frame1 = vol[:, :, f1_idx]
            frame2 = vol[:, :, f2_idx]

            # augmentations (same style as your other datasets)
            angle = random.randint(0, 359)
            frame1 = rotate(frame1, angle, reshape=False, mode='nearest')
            frame2 = rotate(frame2, angle, reshape=False, mode='nearest')

            resize_size = random.randint(256, 300)
            frame1 = resize(frame1, (resize_size, resize_size), anti_aliasing=True).astype(np.float32)
            frame2 = resize(frame2, (resize_size, resize_size), anti_aliasing=True).astype(np.float32)
            frame1, frame2 = get_random_crop([frame1, frame2], 256, 256)

            frame1_input = cap_image(frame1.copy(), random.randint(-400, -200), random.randint(300, 500))
            frame2_input = cap_image(frame2.copy(), random.randint(-400, -200), random.randint(300, 500))

            if np.std(frame1_input) < 10 or np.std(frame2_input) < 10:
                noise = (np.random.normal(loc=1, scale=uniform(0, 0.4), size=frame1_input.shape)) ** 2
                frame1_input += noise
                frame2_input += noise
                frame1 += noise
                frame2 += noise

            frame1_input = gamma_contrast(frame1_input)
            frame2_input = gamma_contrast(frame2_input)

            # [1,H,W]
            frame1_input = frame1_input[np.newaxis, ...]
            frame2_input = frame2_input[np.newaxis, ...]
            frame1 = frame1[np.newaxis, ...]
            frame2 = frame2[np.newaxis, ...]

            return frame1_input, frame2_input, frame1, frame2

        except Exception as e:
            print(f"Error in Dataset_numpy_3d.__getitem__: {e}")
            return None

    def _expand_sources_to_files(self, sources):
        """Turn mixed list of folders/files into a flat list of .npy files."""
        files = []
        for s in sources:
            if os.path.isdir(s):
                fs = sorted(glob.glob(os.path.join(s, "*.npy")))
                files.extend(fs)
            elif os.path.isfile(s) and s.endswith(".npy"):
                files.append(s)
            else:
                # ignore unknown entries but print once for debugging
                print(f"[Dataset_numpy_3d] skipping unknown source: {s}")
        return files

    def _build_index(self, sources):
        rng = np.random.RandomState(self.seed)

        npy_files = self._expand_sources_to_files(sources)
        if len(npy_files) == 0:
            print("[Dataset_numpy_3d] No .npy files found. Index is empty.")
            return

        count = 0
        for n, npy_file in enumerate(npy_files):
            if n % 50 == 0:
                print(f"Indexing {n}/{len(npy_files)}: {npy_file}")

            try:
                vol_dict = np.load(npy_file, allow_pickle=True).item()
                vol = vol_dict.get('image', None)
                if vol is None or not hasattr(vol, "shape"):
                    continue
                vol = vol.astype(np.float32)

                if vol.ndim != 3:
                    continue

                num_slices = vol.shape[-1]
                if num_slices < 4:
                    continue

                interval = int(rng.choice(self.slice_interval_choices))

                # build slice pairs
                for f in range(0, num_slices, interval + 1):
                    if f + interval < num_slices:
                        if rng.rand() < 0.5:
                            temp = {'vol': npy_file, 'f1': f, 'f2': f + interval}
                        else:
                            temp = {'vol': npy_file, 'f1': f + interval, 'f2': f}

                        self.info[count] = temp
                        self.list_IDs.append(count)
                        count += 1

            except Exception as e:
                print(f"[Dataset_numpy_3d] Error loading {npy_file}: {e}")
                continue

        print(f"[Dataset_numpy_3d] Total samples created: {len(self.list_IDs)}")
        self.shuffle_list()

    def shuffle_list(self):
        random.shuffle(self.list_IDs)
    


def split_90_10_per_folder(folders, seed=10):
    rng = np.random.RandomState(seed)
    train_files, val_files = [], []

    for folder in folders:
        files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        idx = np.arange(len(files))
        rng.shuffle(idx)

        n_train = int(0.9 * len(files))
        n_train = max(1, n_train)
        if len(files) - n_train == 0 and len(files) > 1:
            n_train = len(files) - 1

        train_files.extend([files[i] for i in idx[:n_train]])
        val_files.extend([files[i] for i in idx[n_train:]])

        print(f"[{os.path.basename(folder)}] total={len(files)} train={n_train} val={len(files)-n_train}")

    random.shuffle(train_files)
    random.shuffle(val_files)
    print(f"[TOTAL] train={len(train_files)} val={len(val_files)}")
    return train_files, val_files
