import numpy as np
import h5py
from scipy.io import loadmat


def Patch(data,height_index,width_index,PATCH_SIZE):
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    return patch

img = h5py.File('D:\HSI_Projects\hyperspectral_data\Chikusei\Chikusei.mat')['chikusei']
gt = loadmat('D:\HSI_Projects\hyperspectral_data\Chikusei\Chikusei_gt.mat')['GT'][0][0][0]
img = np.array(img).transpose((2, 1, 0))
print(img.shape)

print(gt.max()+1)




img = img[:, :, 0:100]
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
del img_temp

gt_temp = np.zeros((m + 2 * PATCH_SIZE, n + 2 * PATCH_SIZE), dtype=np.int8)
gt_temp[PATCH_SIZE:(m + PATCH_SIZE), PATCH_SIZE:(n + PATCH_SIZE)] = gt[:, :]
gt = gt_temp

[m, n, b] = img.shape


def preparation():

    data = []
    label = []

    for i in range(PATCH_SIZE, m - PATCH_SIZE):
        for j in range(PATCH_SIZE, n - PATCH_SIZE):
            if gt[i, j] == 0:
                continue
            else:
                # count += 1
                temp_data = Patch(img, i, j, PATCH_SIZE)
                temp_label = gt[i, j] - 1

                data.append(temp_data) 
                label.append(temp_label)


    
    

    data = np.asarray(data)
    print(data.shape)  
    data = np.squeeze(data)
    print("squeeze : ", data.shape)
    label = np.asarray(label)
    print(label.shape)  
    label = np.squeeze(label)
    print("squeeze : ", label.shape)


    
    indices = np.arange(data.shape[0])
    shuffled_indices = np.random.permutation(indices)
    images = data[shuffled_indices]
    labels = label[shuffled_indices]  # 打乱顺序

    y = labels

    n_classes = y.max() + 1
    t_labeled = []

    # 改变类别数量---------------------------------------------------
    for c in range(n_classes-1): 
        i = indices[y == c][:200]
        t_labeled += list(i)


    # 将其划分分训练和检验两个数据集
    t_images = images[t_labeled]
    print('t_images', t_images.shape)
    t_labels = labels[t_labeled]
    print('t_labels', t_labels.shape)

    # 训练数据集
    f = h5py.File(r'./data/source_data/CH-' + str(PATCH_SIZE*2) + '-' + str(PATCH_SIZE*2) + '-100.h5', 'w')  # 每类200个
    f['data'] = t_images
    f['label'] = t_labels
    f.close()



preparation()
