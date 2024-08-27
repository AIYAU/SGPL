import numpy as np
import h5py
from scipy.io import loadmat
from functools import reduce

def Patch(data,height_index,width_index,PATCH_SIZE): 
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    return patch

img = loadmat('D:\HSI_Projects\hyperspectral_data\KSC\KSC.mat')['KSC']
gt = loadmat('D:\HSI_Projects\hyperspectral_data\KSC\KSC_gt.mat')['KSC_gt']
print(img.shape) 
# 统计每类样本所含个数
dict_k = {}
for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
        if gt[i][j] in range(0,gt.max()+1):
            if gt[i][j] not in dict_k:
                dict_k[gt[i][j]]=0
            dict_k[gt[i][j]] +=1

print(dict_k)

img = img[:, :, :100]
img = (img * 1.0 - img.min()) / (img.max() - img.min())

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

label_num = gt.max() 
data = []
label = []

count = 0 


for i in range(0, m):
    for j in range(0, n):
        if gt[i, j] == 0:
            continue
        else:
            count += 1
            temp_data = Patch(img, i, j, PATCH_SIZE)
            temp_label = np.zeros((1, label_num), dtype=np.int8)  
            temp_label = gt[i, j] - 1
            data.append(temp_data)             
            label.append(temp_label)
            

print(count)
data = np.array(data)
data = np.squeeze(data)
print(data.shape) 

label = np.array(label)
print(label.shape)
label = np.squeeze(label)
print("squeeze : ",label.shape) 



indices = np.arange(data.shape[0]) 
shuffled_indices = np.random.permutation(indices)
images = data[shuffled_indices]
labels = label[shuffled_indices]  # 打乱顺序
y = labels 
print(y)


s_labeled = []


sample = {'8': 431, '12': 503, '13': 927, '11': 419, '5': 161, '1': 761, '4': 252, '6': 229, '2': 243, '3': 256, '10': 404, '7': 105, '9': 520}
for c in range(y.max() + 1): 
    if sample[str(c + 1)] < 200:
        pass
    else:
        i = indices[y == c][:200]
        s_labeled += list(i) 



s_images = images[s_labeled]
print('s_images', s_images.shape)
s_labels = labels[s_labeled]
print('s_labels', s_labels.shape)

f = h5py.File(r'./data/source_data/KSC-' + str(PATCH_SIZE*2) + '-' + str(PATCH_SIZE*2) + '-100.h5', 'w') # 每类200个
f['data'] = s_images 
f['label'] = s_labels 
f.close()