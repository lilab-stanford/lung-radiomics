import zipfile
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import zipfile
from pydicom import dcmread
from pydicom.sequence import Sequence
import glob
import skimage
from get_lung_mask import lung_mask
import os
from scipy.ndimage.measurements import label
import warnings
warnings.filterwarnings("ignore")
import nibabel as nib 
import numpy as np 
from sklearn.cluster import KMeans
 

def remove_noise(seg_volume, label_volume):
    label_volume, label_n = label(label_volume)
    if label_n > 1:
        max_size = 0
        max_label = 0
        for i in range(1, label_n + 1):
            if np.sum(label_volume == i) > max_size:
                max_size = np.sum(label_volume == i)
                max_label = i
        # print('max_size:', max_size, 'max_label:', max_label)
        for i in range(1, label_n + 1):
            if i != max_label:
                seg_volume[label_volume == i] = 0
    return seg_volume

def zip_extract_dicom_sequences(zip_path):
    sequences = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取ZIP文件中的所有文件名称
        file_list = zip_ref.namelist()
        dicom_files = [file_name for file_name in file_list if file_name.endswith('.dcm')]
        
        # 按序列分组
        for dicom_file in dicom_files:
            with zip_ref.open(dicom_file, 'r') as file:
                dcm = dcmread(file)
                # 检查是否已经在某个序列中
                found_in_sequence = False
                for seq in sequences:
                    if seq[0] == dcm.SeriesInstanceUID:
                        seq[1].append(dcm)
                        found_in_sequence = True
                        break
                if not found_in_sequence:
                    # 创建新的序列
                    sequences.append([dcm.SeriesInstanceUID, [dcm]])
    return sequences

def respacing(ct_image, mask, original_spacing, new_spacing):
    # z, x, y
    new_shape = np.round(np.array(ct_image.shape) * np.array(original_spacing) / np.array(new_spacing)).astype(int)
    #ct_image = (((ct_image-np.amin(ct_image))*255)/(np.amax(ct_image)-np.amin(ct_image)))
    ct_image = skimage.transform.resize(ct_image, new_shape, 1)
    new_mask = skimage.transform.resize(mask, new_shape, 0, anti_aliasing=False)
    ct_image = (((ct_image-np.amin(ct_image))/(np.amax(ct_image)-np.amin(ct_image))) * 255).astype('int16')
    new_mask = ((new_mask-np.amin(new_mask))*1/(np.amax(new_mask)-np.amin(new_mask))).astype('int16')
    return ct_image, new_mask

def save_nii(array, save_path):
    dcm = sitk.GetImageFromArray(array)
    sitk.WriteImage(dcm, save_path)

def nibabel_nii(array, save_path):
    new_image = nib.Nifti1Image(array, np.eye(4)) 
    nib.save(new_image, save_path)


if __name__ == '__main__':
    zip_files = glob.glob('/storage1/jiaorushi/20240116Rushi/MD/Simulation NSCLC DICOM/'+'/*.zip')
    save_path = '/storage1/jiaorushi/20240116Rushi/MD/preprocessed/'
    save_path2 = '/storage1/jiaorushi/20240116Rushi/MD/preprocessed2/'
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    zip_num = 0
    seq_num = 0
    for zip_file_path in zip_files:
        zip_num += 1
        save_file = save_path + zip_file_path.split('/')[-1].split('.')[0]
        save_img = save_file + '_image.nii.gz'
        save_file2 = save_path2 + zip_file_path.split('/')[-1].split('.')[0]
        save_img2 = save_file2 + '_image.nii.gz'
        if not os.path.exists(save_img):
            print('not exists', save_img)
            sequences = zip_extract_dicom_sequences(zip_file_path)
            #for seq in sequences:
            if 1:
                seq = sequences[0]
                try:
                #if 1:
                    if 1:
                    #if seq[1][0].SeriesDescription == '90.0%':
                        seq_num += 1
                        sorted_datasets = sorted(seq[1], key=lambda x: float(x[0x0020, 0x0032][-1]), reverse=True)
                        array = np.empty((0,) + sorted_datasets[0].pixel_array.shape, dtype=sorted_datasets[0].pixel_array.dtype)
                        for ds in sorted_datasets:
                            array = np.vstack((array, ds.pixel_array[np.newaxis]))
                        z = ds[0x0018, 0x0088].value
                        xy = list(ds[0x0028, 0x0030])
                        xy.insert(0, abs(z))
                        xy = [float(s) for s in xy]

                        mask_all = []
                        
                        middle = array[array.shape[0]//2].copy()[100:400,100:400]  
                        mean = np.mean(middle)
                        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
                        centers = sorted(kmeans.cluster_centers_.flatten())
                        threshold = np.mean(centers)  
                        for image in array:
                            mask, th = lung_mask(image.copy(), threshold)
                            mask_all.append(mask)
                            print(mask.shape)
                        mask_all = np.array(mask_all)
                        # mask_all[:mask_all.shape[0]//4] = 0
                        mask_all[-mask_all.shape[0]//4:] = 0
                        new_array, new_mask_all = respacing(array, mask_all, xy, [1,1,1])
                        print(save_file, array.shape, xy, new_array.shape, new_mask_all.shape)
                        save_img2 = save_file2 + '_image.nii.gz'
                        save_mask2 = save_file2 + '_mask.nii.gz'
                        print(np.amax(new_array), np.amax(new_mask_all))
                        save_nii(new_array, save_img2)
                        save_nii(new_mask_all, save_mask2)
                except:
                    print(save_file)
    print(zip_num, seq_num)







