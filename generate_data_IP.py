import numpy as np
import h5py
from scipy.io import loadmat


def Patch(data,height_index,width_index,PATCH_SIZE): 
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])                  
    return patch


seed_number = 7
np.random.seed(int(seed_number))

img = loadmat('D:\HSI_Projects\hyperspectral_data\Indian_pines\Indian_pines_corrected.mat')['indian_pines_corrected']
gt = loadmat('D:\HSI_Projects\hyperspectral_data\Indian_pines\Indian_pines_gt.mat')['indian_pines_gt']

# 只选取前100个波段
img = img[:, :, 0:100]


# 归一化
img = ( img * 1.0 - img.min() ) / ( img.max() - img.min() )
[m, n, b] = img.shape
label_num = gt.max()  
PATCH_SIZE = 14   

# padding the hyperspectral images
img_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE, b), dtype=np.float32)
img_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE), :] = img[:, :, :]

for i in range(PATCH_SIZE):
    img_temp[i, :, :] = img_temp[2 * PATCH_SIZE - i, :, :]
    img_temp[m + PATCH_SIZE + i, :, :] = img_temp[m + PATCH_SIZE - i - 2, :, :]

for i in range(PATCH_SIZE):
    img_temp[:, i, :] = img_temp[:, 2 * PATCH_SIZE - i, :]
    img_temp[:, n + PATCH_SIZE + i, :] = img_temp[:, n + PATCH_SIZE  - i - 2, :]

img = img_temp

gt_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE), dtype=np.int8)
gt_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE)] = gt[:, :]
gt = gt_temp

[m, n, b] = img.shape



def preparation():

    f = open('./data/IP/gt_index_IP.txt', 'w')
    f1 = open('./data/IP/IP_label.txt', 'w')
    data = []
    label = []

    for i in range(PATCH_SIZE, m - PATCH_SIZE):
        for j in range(PATCH_SIZE, n - PATCH_SIZE):
            if gt[i, j] == 0:
                continue
            else:
                temp_data = Patch(img, i, j, PATCH_SIZE)
                temp_label = gt[i, j] - 1 
                data.append(temp_data)  
                label.append(temp_label)
                gt_index = ((i - PATCH_SIZE) * 217 + j - PATCH_SIZE) 
                f.write(str(gt_index) + '\n')
                f1.write(str(temp_label) + '\n')

    data = np.array(data)
    print(data.shape)  
    data = np.squeeze(data)
    print("squeeze : ", data.shape) 
    label = np.array(label)
    print(label.shape)  
    label = np.squeeze(label)
    print("squeeze : ", label.shape)
    print(np.unique(label)) 
   
    f = h5py.File(r'.\data\IP\IP_' + str(PATCH_SIZE*2) + '_' + str(PATCH_SIZE*2) + '_100_test.h5', 'w')
    f['data'] = data
    f['label'] = label
    f.close()

    # 每类随机采样num_s个生成支撑样本集
    num_s = 5  # 支撑样本集数量
    indices = np.arange(data.shape[0]) 
    shuffled_indices = np.random.permutation(indices)
    data = data[shuffled_indices]
    label = label[shuffled_indices]
    data_s = []
    label_s = []

    for i in range(label_num): 
        count = 0
        for j in range(10249): # 数量循环
            if label[j] == i and count <= num_s-1: # 如果标记为第i类
                data_s.append(data[j, :])
                label_s.append(label[j])
                count += 1
    data_s = np.array(data_s)
    label_s = np.array(label_s)
    print(data_s.shape)
    print(np.unique(label_s))
    print(label_s.shape)

    PATH = './data/IP/IP_' + str(PATCH_SIZE*2) + '_' + str(PATCH_SIZE*2) + '_100_support' + str(num_s) + '.h5'
    f = h5py.File(PATH, 'w')
    f['data_s'] = data_s
    f['label_s'] = label_s 
    f.close()


preparation()