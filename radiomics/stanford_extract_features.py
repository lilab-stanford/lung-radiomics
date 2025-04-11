import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage
import SimpleITK as sitk
import cv2
import six
import radiomics
import csv
import pandas as pd
from radiomics import featureextractor  # This module is used for interaction with pyradiomics





def avg_head(att):
    att = np.mean(att, axis=0).squeeze()
    att = np.mean(att, axis=0).squeeze()
    return att

def clip(array, amax=255, amin=200):
    array[array>amax] = amax
    array[array<amin] = amin
    array = (array - np.min(array))/(np.max(array)-np.min(array)) * 255
    return array

def catch_features(image,mask):
    # if imagePath is None or maskPath is None:  # Something went wrong, in this case PyRadiomics will also log an error
    #     raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
    settings = {}
    # settings['binWidth'] = 25  # 5
    # settings['sigma'] = [3, 5]
    # settings['Interpolator'] = sitk.sitkBSpline
    # settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
    # settings['voxelArrayShift'] = 1000  # 300
    # settings['normalize'] = True
    # settings['normalizeScale'] = 100
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    #extractor = featureextractor.RadiomicsFeatureExtractor()
    print('Extraction parameters:\n\t', extractor.settings)

    extractor.enableImageTypeByName('Original')
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation', 'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
    extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion','Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn','Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness'])
    extractor.enableFeaturesByName(GLCM=['SumVariance', 'Dissimilarity'])
    extractor.enableFeaturesByName(GLDM=['Dependencepercentage', 'GrayLevelNon-UniformityNormalized'])
# 上边两句我将一阶特征和形状特征中的默认禁用的特征都手动启用，为了之后特征筛选
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    feature_cur = []
    feature_name = []
    result = extractor.execute(image, mask, label=1)
    for key, value in six.iteritems(result):
        feature_name.append(key)
        feature_cur.append(value)
#     print(len(feature_cur[37:]))
#     name = feature_name[37:]
#     name = np.array(name)

#     for i in range(len(feature_cur[37:])):
#         feature_cur[i+37] = float(feature_cur[i+37])
    return feature_cur, feature_name

    

if __name__ == '__main__':
    #files = glob.glob('/storage1/jiaorushi/20240116Rushi/Stanford/preprocessed/*image.nii.gz') 
    nums = list(range(1, 140)) 
    csv_save = 'stanford_features_140.csv'  
    csv_file = open(csv_save, 'w')
    writer = csv.writer(csv_file)
    f = 0
    a = time.time()
    #for file in files[:3]:
    for num in nums[70:]:
        file = '/storage1/jiaorushi/20240116Rushi/Stanford/preprocessed/' + str(num) + '_image.nii.gz'
        if os.path.exists(file):
            try:
                image_file = file
                mask_file = '_'.join(file.split('_')[:-1]) + '_mask.nii.gz'
                print(mask_file)
                img = sitk.ReadImage(image_file)
                mask = sitk.ReadImage(mask_file)
                print(sitk.GetArrayFromImage(img).shape, sitk.GetArrayFromImage(mask).shape)
                features, name = catch_features(img, mask)
                if f == 0:
                    name.insert(0, 'PID')
                    writer.writerow(name)
                f = f+1
                features.insert(0, file)
                writer.writerow(features)
            
            except:
                print('error!!!!!!!!', file)          
    csv_file.close()
    print(time.time()-a)
                
            
            
            
            
        

