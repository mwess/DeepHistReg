import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import random
import SimpleITK as sitk

import torch
import torch.utils as utils
import pandas as pd
from skimage import io
import cv2


def collate_to_list_unsupervised(batch):
    sources = [item[0].view(item[0].size(0), item[0].size(1)) for item in batch]
    targets = [item[1].view(item[1].size(0), item[1].size(1)) for item in batch]
    return sources, targets

def collate_to_list_segmentation(batch):
    sources = [item[0].view(item[0].size(0), item[0].size(1)) for item in batch]
    targets = [item[1].view(item[1].size(0), item[1].size(1)) for item in batch]
    source_masks = [item[2].view(item[2].size(0), item[2].size(1)) for item in batch]
    target_masks = [item[3].view(item[3].size(0), item[3].size(1)) for item in batch]
    return sources, targets, source_masks, target_masks

class UnsupervisedLoader(utils.data.Dataset):
    def __init__(self, data_path, transforms=None, randomly_swap=False):
        self.data_path = data_path
        self.table = pd.read_csv(data_path)
        #self.all_ids = os.listdir(self.data_path)
        self.transforms = transforms
        self.randomly_swap = randomly_swap

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        source_image_path = row['source_image']
        source_mask_path = row['source_mask']
        target_image_path = row['target_image']
        target_mask_path = row['target_mask']
        source = io.imread(source_image_path, as_gray=True)
        target = io.imread(target_image_path, as_gray=True)
        source_mask = io.imread(source_mask_path)
        target_mask = io.imread(target_mask_path)
        source = nd.gaussian_filter(source, 0.75)
        target = nd.gaussian_filter(target, 0.75)
        source = source * source_mask
        target = target * target_mask
        #case_id = self.all_ids[idx]
        #source_path = os.path.join(self.data_path, str(case_id), "source.mha")
        #target_path = os.path.join(self.data_path, str(case_id), "target.mha")
        #source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
        #target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))

        if self.transforms is not None:
            source, target, _ = self.transforms(source, target)

        if self.randomly_swap:
            if random.random() > 0.5:
                pass
            else:
                source, target = target, source
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))
        source_tensor, target_tensor = torch.from_numpy(source.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        return source_tensor, target_tensor

#class UnsupervisedLoader(utils.data.Dataset):
#    def __init__(self, data_path, transforms=None, randomly_swap=False):
#        self.data_path = data_path
#        self.all_ids = os.listdir(self.data_path)
#        self.transforms = transforms
#        self.randomly_swap = randomly_swap
#
#    def __len__(self):
#        return len(self.all_ids)
#
#    def __getitem__(self, idx):
#        case_id = self.all_ids[idx]
#        source_path = os.path.join(self.data_path, str(case_id), "source.mha")
#        target_path = os.path.join(self.data_path, str(case_id), "target.mha")
#        source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
#        target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
#
#        if self.transforms is not None:
#            source, target, _ = self.transforms(source, target)
#
#        if self.randomly_swap:
#            if random.random() > 0.5:
#                pass
#            else:
#                source, target = target, source
#
#        source_tensor, target_tensor = torch.from_numpy(source.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
#        return source_tensor, target_tensor

class SegmentationLoader(utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_ids = os.listdir(self.data_path)

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        case_id = self.all_ids[idx]
        source_path = os.path.join(self.data_path, str(case_id), "source.mha")
        target_path = os.path.join(self.data_path, str(case_id), "target.mha")
        source_mask_path = os.path.join(self.data_path, str(case_id), "source_mask.mha")
        target_mask_path = os.path.join(self.data_path, str(case_id), "target_mask.mha")
        source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        source_mask = sitk.GetArrayFromImage(sitk.ReadImage(source_mask_path))
        target_mask = sitk.GetArrayFromImage(sitk.ReadImage(target_mask_path))

        source_tensor, target_tensor = torch.from_numpy(source.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        source_mask_tensor, target_mask_tensor = torch.from_numpy(source_mask.astype(np.float32)), torch.from_numpy(target_mask.astype(np.float32))
        return source_tensor, target_tensor, source_mask_tensor, target_mask_tensor