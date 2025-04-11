import numpy as np
from scipy.stats import entropy

def calculate_2d_entropy(image, direction='horizontal'):
    """
    计算二维熵（2D Entropy），用于衡量肿瘤中血管分布的疏密程度
    
    参数:
    image: 2D numpy数组，表示灰度图像（肿瘤区域的血管图像）
    direction: 计算GLCM的方向，可选 'horizontal', 'vertical', 'diagonal'
    
    返回:
    entropy_2d: 二维熵的值
    """
    # 确保输入是2D数组
    if len(image.shape) != 2:
        raise ValueError("输入必须是一个2D灰度图像")
    
    # 初始化灰度共生矩阵（GLCM）
    glcm = np.zeros((256, 256), dtype=int)
    
    # 根据方向计算GLCM
    if direction == 'horizontal':
        # 水平方向：当前像素和右侧像素
        for i in range(image.shape[0]):
            for j in range(image.shape[1] - 1):
                current_pixel = image[i, j]
                next_pixel = image[i, j + 1]
                glcm[current_pixel, next_pixel] += 1
    elif direction == 'vertical':
        # 垂直方向：当前像素和下方像素
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1]):
                current_pixel = image[i, j]
                next_pixel = image[i + 1, j]
                glcm[current_pixel, next_pixel] += 1
    elif direction == 'diagonal':
        # 对角线方向：当前像素和右下方像素
        for i in range(image.shape[0] - 1):
            for j in range(image.shape[1] - 1):
                current_pixel = image[i, j]
                next_pixel = image[i + 1, j + 1]
                glcm[current_pixel, next_pixel] += 1
    else:
        raise ValueError("方向参数必须是 'horizontal', 'vertical' 或 'diagonal'")
    
    # 将GLCM归一化为概率分布
    glcm = glcm / glcm.sum()
    
    # 计算二维熵
    entropy_2d = entropy(glcm.flatten())
    
    return entropy_2d

# 示例使用
if __name__ == "__main__":
    # 创建一个示例灰度图像（8位深度，0-255）
    # 假设这是一个肿瘤区域的血管图像
    image = np.array([
        [50, 100, 150, 200],
        [100, 150, 200, 250],
        [150, 200, 250, 50],
        [200, 250, 50, 100]
    ], dtype=np.uint8)
    
    # 计算二维熵
    entropy_value = calculate_2d_entropy(image, direction='horizontal')
    print(f"二维熵（水平方向）: {entropy_value}")