import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob


def lung_mask(img, threshold = None):
    #提取肺部大致均值
    middle = img[100:400,100:400]  
    mean = np.mean(middle)  

    # # 将图片最大值和最小值替换为肺部大致均值
    # max = np.max(img)
    # min = np.min(img)
    # img[img==max]=mean
    # img[img==min]=mean
    # image_array = img
    if threshold is None:
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)  
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    # 聚类完成后，清晰可见偏黑色区域为一类，偏灰色区域为另一类。
    image_array = thresh_img
    
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))  
    dilation = morphology.dilation(eroded,np.ones([10,10]))  
    labels = measure.label(dilation) 

    label_vals = np.unique(labels)
    regions = measure.regionprops(labels) # 获取连通区域

    # 设置经验值，获取肺部标签
    good_labels = []
    for prop in regions:
        area = prop.area
        if 1:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
    # 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    return mask, threshold    
