from skimage import io
import SimpleITK as sitk
import skimage.color as color
import torch
import numpy as np
import glob
import os
import cv2
import pandas as pd
import scipy.ndimage as nd

import dataloaders as dl
from rotation_alignment import rotation_alignment
import utils

def get_data_paths(with_segmentation=False):
    #root_dir = '/mnt/work/workbench/maximilw/wsi/exports/multiomics/'
    root_dir = '/home/data/exports/multiomics/'
    he_path = os.path.join(root_dir, 'spatial_transcriptomics/')
    hes_path = os.path.join(root_dir, 'hes_scans/')
    lps_path = os.path.join(root_dir, 'ihc/lps')
    lta_path = os.path.join(root_dir, 'ihc/lta')
    mts_path = os.path.join(root_dir, 'mts')
    hes_peptides_path = os.path.join(root_dir, 'hes_scans/peptides')
    he_dataset =  get_data_subset(he_path, with_segmentation)
    hes_dataset = get_data_subset(hes_path, with_segmentation)
    lps_dataset = get_data_subset(lps_path, with_segmentation)
    lta_dataset = get_data_subset(lta_path, with_segmentation)
    mts_dataset = get_data_subset(mts_path, with_segmentation)
    hes_peptides_dataset = get_data_subset(hes_peptides_path, with_segmentation)
    hes_peptides_dataset = [x for x in hes_peptides_dataset if not 'MP_F13' in x[0]]
    dataset = he_dataset + hes_dataset + lps_dataset + lta_dataset + mts_dataset + hes_peptides_dataset
    return dataset

def get_image_paths(root_dir):
    image_subdir = 'segmented_images'
    return glob.glob(os.path.join(root_dir, image_subdir, '*.tif'))

def get_dataset():
    data_paths = get_data_paths()
    image_paths = get_data_subset(data_paths)
    return image_paths
    

def get_data_subset_with_segmentation(root_dir):
    image_subdir = 'segmented_images'
    mask_subdir = 'masks'
    # he_root = '/mnt/work/workbench/maximilw/wsi/exports/multiomics/spatial_transcriptomics/'#segmented_images/*.tif'
    image_data = []
    image_paths = glob.glob(os.path.join(root_dir, image_subdir, '*.tif'))
    for image_path in image_paths:
        mask_path = image_path.replace(image_subdir, mask_subdir)
        image_data.append((image_path, mask_path))
    return image_data

def get_data_subset(root_dir, with_segmentation=False):
    if with_segmentation:
        return get_data_subset_with_segmentation(root_dir)
    image_subdir = 'segmented_images'
    paths =  glob.glob(os.path.join(root_dir, image_subdir, '*.tif'))
    return [[x] for x in paths]

def get_dataset_with_segmentation():
    he_dataset = get_data_subset_with_segmentation('/mnt/work/workbench/maximilw/wsi/exports/multiomics/spatial_transcriptomics/')
    hes_dataset = get_data_subset_with_segmentation('/mnt/work/workbench/maximilw/wsi/exports/multiomics/hes_scans')
    lps_dataset = get_data_subset_with_segmentation('/mnt/work/workbench/maximilw/wsi/exports/multiomics/ihc/lps')
    lta_dataset = get_data_subset_with_segmentation('/mnt/work/workbench/maximilw/wsi/exports/multiomics/ihc/lta')
    mts_dataset = get_data_subset_with_segmentation('/mnt/work/workbench/maximilw/wsi/exports/multiomics/mts')
    hes_peptides_dataset = get_data_subset_with_segmentation('/mnt/work/workbench/maximilw/wsi/exports/multiomics/hes_scans/peptides')
    hes_peptides_dataset = [x for x in hes_peptides_dataset if not 'MP_F13' in x[0]]
    dataset = he_dataset + hes_dataset + lps_dataset + lta_dataset + hes_peptides_dataset + mts_dataset
    return dataset    

def ordered_images_to_batches(image_list, with_segmentation=False, is_cuda=False):
    batches = []
    for i in range(len(image_list) - 1):
        if with_segmentation:
            batch = load_image_pair_with_segmentation(image_list[i], image_list[i+1], is_cuda)
        else:
            batch = load_image_pair(image_list[i], image_list[i+1], is_cuda)
        batches.append(batch)
    return batches    

def load_image_pair(source_paths, target_paths, is_cuda=False):
    source_path = source_paths[0]
    target_path = target_paths[0]
    #resolution = (512, 512)
    #source = io.imread(source_path, as_gray=True).astype(np.float32)
    #target = io.imread(target_path, as_gray=True).astype(np.float32)
    #source = cv2.resize(source, resolution)
    #target = cv2.resize(target, resolution)
    source = load_image(source_path)
    target = load_image(target_path)
    source_tensor = torch.from_numpy(source.astype(np.float32))
    target_tensor = torch.from_numpy(target.astype(np.float32))
    if is_cuda:
        source_tensor = source_tensor.cuda()
        target_tensor = target_tensor.cuda()
    return {
            'source_path': source_path,
            'source_image': source_tensor, 
            'target_path': target_path,
            'target_image': target_tensor
            }

def load_image_pair_with_segmentation(source_paths, target_paths, is_cuda=False):
    source, source_mask = load_image_with_mask(source_paths)
    target, target_mask = load_image_with_mask(target_paths)
    source = source * source_mask
    target = target * target_mask
    source_tensor = torch.from_numpy(source.astype(np.float32))
    target_tensor = torch.from_numpy(target.astype(np.float32))
    if is_cuda:
        source_tensor = source_tensor.cuda()
        target_tensor = target_tensor.cuda()
    return {
            'source_path': source_paths[0],
            'source_image': source_tensor, 
            'target_path': target_paths[0],
            'target_image': target_tensor
            }



def load_image(path):
    #print(path)
    resolution = (512, 512)
    output_max_size = 1024
    image = color.rgb2gray(sitk.GetArrayFromImage(sitk.ReadImage(path)))
    #image = io.imread(path, as_gray=True).astype(np.float32)
    image = cv2.resize(image, resolution)
    image = 1 - utils.normalize(image)
    #print(image.min(), image.max(), image.shape)
    resample_factor = np.max(image.shape) / output_max_size
    gaussian_sigma = resample_factor / 1.25
    smoothed_image = nd.gaussian_filter(image, gaussian_sigma)
    resampled_image = utils.resample_image(smoothed_image, resample_factor)
    resampled_image = utils.normalize(resampled_image)
    return resampled_image

def load_image_with_mask(paths):
    #print(path)
    image_path, mask_path = paths
    resolution = (512, 512)
    max_output_size = 512
    #print(mask_path)
    image = color.rgb2gray(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))
    mask = io.imread(mask_path).astype(np.ubyte)
    #print(mask)

    #image = io.imread(path, as_gray=True).astype(np.float32)
    image = cv2.resize(image, resolution)
    mask = cv2.resize(mask, resolution)
    image = 1 - utils.normalize(image)
    mask = mask / np.max(mask)
    #print(np.isnan(mask).sum())
    #print(image.min(), image.max(), image.shape)
    resample_factor = np.max(image.shape) / max_output_size
    gaussian_sigma = resample_factor / 1.25
    smoothed_image = nd.gaussian_filter(image, gaussian_sigma)
    resampled_image = utils.resample_image(smoothed_image, resample_factor)
    resampled_image = utils.normalize(resampled_image)
    resampled_mask = (utils.resample_image(mask, resample_factor) > 0.5).astype(np.ubyte)
    return resampled_image, resampled_mask



def partition_multiomics_dataset_with_segmentation(dataset):
    partitioned_dataset = {}
    for paths in dataset:
        name = paths[0].rsplit('/', maxsplit=1)[-1]
        pref = name[:3]
        suff = name.rsplit('.', maxsplit=1)[0][-2:]
        key = pref + '_' + suff
        if key not in partitioned_dataset:
            partitioned_dataset[key] = []
        partitioned_dataset[key].append(paths)
    for key in partitioned_dataset:
        partitioned_dataset[key] = sorted(partitioned_dataset[key], key=lambda x: get_section_position(x[0]))
    return partitioned_dataset



def partition_multiomics_dataset(dataset, with_segmentation=False):
    if with_segmentation:
        return partition_multiomics_dataset_with_segmentation(dataset)
    partitioned_dataset = {}
    for data in dataset:
        print(f'data: {data}')
        fpath = data[0]
        name = fpath.rsplit('/', maxsplit=1)[-1]
        pref = name[:3]
        suff = name.rsplit('.', maxsplit=1)[0][-2:]
        key = pref + '_' + suff
        if key not in partitioned_dataset:
            partitioned_dataset[key] = []
        partitioned_dataset[key].append(data)
    
    for key in partitioned_dataset:
        partitioned_dataset[key] = sorted(partitioned_dataset[key], key=lambda x: get_section_position(x[0]))
        
    return partitioned_dataset   

def get_section_position(fpath):
    fname = fpath.rsplit('/', maxsplit=1)[-1]
    if fname[3:].startswith('_ST_' ):
        return 2
    if fname[3:].startswith(' HE R1_'):
        return 1
    if fname[3:].startswith(' HE R11_'):
        return 11
    if fname[3:].startswith(' HE R21_'):
        return 21
    if fname[3:].startswith(' M- V7'):
        return 7
    if fname[3:].startswith(' M+ V6'):
        return 6
    if fname[4:].startswith('LTA'):
        return 10
    if fname[4:].startswith('LPS'):
        return 9
    if fname[4:].startswith('MP_F3'):
        return 3
    if fname.split()[1].startswith('MM'):
        return 8
    print(fname)

def get_data_subset_with_segmentation(root_dir):
        image_subdir = 'segmented_images'
        mask_subdir = 'masks'
        # he_root = '/mnt/work/workbench/maximilw/wsi/exports/multiomics/spatial_transcriptomics/'#segmented_images/*.tif'
        image_data = []
        image_paths = glob.glob(os.path.join(root_dir, image_subdir, '*.tif'))
        for image_path in image_paths:
            mask_path = image_path.replace(image_subdir, mask_subdir)
            image_data.append((image_path, mask_path))
        return image_data

def prealignment(params):
    training_path = params['training_path']
    validation_path = params['validation_path']
    training_loader = dl.UnsupervisedLoader(training_path, transforms=None)
    validation_loader = dl.UnsupervisedLoader(validation_path, transforms=None) 
    training_dataloader = torch.utils.data.DataLoader(training_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)
    validation_dataloader = torch.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)

#dataset = get_dataset_with_segmentation()
dataset = get_data_paths(with_segmentation=True)
cores = partition_multiomics_dataset(dataset, with_segmentation=True)
#print(cores)
#exit()
windows_sizes = [9, 16, 32, 64, 128, 196, 256, 384, 512]
#windows_sizes = [512]
base_columns = ['batch', 'pair', 'source_name', 'target_name']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
is_cuda = torch.cuda.is_available()
print(device)
for window_size in windows_sizes:
    base_columns.append(f'cost_{window_size}')
    base_columns.append(f'angle_{window_size}')
table = pd.DataFrame(columns=base_columns)
params = {'angle_step': 1, 'global': True, 'device': device}
for outer_idx, key in enumerate(cores):
    params['window_size'] = -1
    #print(cores[key])
    batch = ordered_images_to_batches(cores[key], is_cuda=is_cuda, with_segmentation=True)
    #print('batch')
    #print(batch)
    for idx, pair in enumerate(batch):
        params['global'] = True
        source_name = pair['source_path'].rsplit('/', maxsplit=1)[-1]
        target_name = pair['target_path'].rsplit('/', maxsplit=1)[-1]
        _, cost_global, angle_global = rotation_alignment(pair['source_image'], pair['target_image'], params=params, device=device)
        cost_global = cost_global.item()
        row = {'batch': key, 'pair': idx + 1, 'angle_global': angle_global, 'cost_global': cost_global, 'source_name': source_name, 'target_name': target_name}
        for window_size in windows_sizes:
            params['window_size'] = window_size
            params['global'] = False
            #print(params)
            _, cost_, angle_ = rotation_alignment(pair['source_image'], pair['target_image'], params=params, device=device)
            cost_ = cost_.item()
            row[f'cost_{window_size}'] = cost_
            row[f'angle_{window_size}'] = angle_


        #scores.append(cost)
        #angles.append(angle)
        #print(source_name, target_name)
        table = table.append(row, ignore_index=True)
        msg = f'No batch: {outer_idx} named: {key} with pair no: {idx + 1} has angle: {angle_global}, cost: {cost_global}'
        print(msg)

#table.to_csv('prealignment_segm_scores.csv', index=False)
table_name = 'prealignment_segm_scores3.csv'
print(f'Writing table {table_name} to file.')
table.to_csv(table_name, index=False)
print(table)
