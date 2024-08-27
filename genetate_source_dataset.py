import h5py
import numpy as np



f=h5py.File('./data/source_data/BO-28-28-100.h5','r')
data1=f['data'][:]
f.close()


f=h5py.File('./data/source_data/HS-28-28-100.h5','r')
data2=f['data'][:]
f.close()


f=h5py.File('./data/source_data/CH-28-28-100.h5','r')
data3=f['data'][:]
f.close()


f=h5py.File('./data/source_data/KSC-28-28-100.h5','r')
data4=f['data'][:]
f.close()


print(data1.shape) 
print(data2.shape) 
print(data3.shape)
print(data4.shape) 

data=np.vstack((data1,data2,data3,data4))
print(data.shape) 


f=h5py.File('./data/source_data/meta_train_' + str(data.shape[0]) + '_' + str(data.shape[1]) + '.h5','w')
f['data']=data
f.close()
