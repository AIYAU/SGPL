import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time

torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-w","--n_way",type = int, default = 20)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 1)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 19)       # query set per class
parser.add_argument("-e","--episode",type = int, default= 10000)
#-----------------------------------------------------------------------------------#
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()



n_way = args.n_way
n_shot = args.n_shot
n_query = args.n_query
EPISODE = args.episode
#-----------------------------------------------------------------------------------#
LEARNING_RATE = args.learning_rate
GPU = args.gpu

n_examples = 200  # 训练数据集中每类200个样本
im_width, im_height, depth = 28, 28, 100 


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(1,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))

        self.layer2 = nn.Sequential(
                        nn.Conv3d(16,32,kernel_size=3,padding=1),
                        nn.BatchNorm3d(32),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        self.layer3 = nn.Sequential(
                        nn.Conv3d(32,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        self.layer4 = nn.Sequential(
                        nn.Conv3d(64,128,kernel_size=3,padding=1),
                        nn.BatchNorm3d(128),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))       
        self.layer5 = nn.Sequential(
                        nn.Conv3d(128,64,kernel_size=(1, 3, 3),padding=0),
                        nn.BatchNorm3d(64),
                        nn.ReLU()
        )


    def forward(self,x):

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = self.layer5(out4)

        return out




class PrototypeGenerator(nn.Module):
    def __init__(self):
        super(PrototypeGenerator, self).__init__()
        self.fc1 = nn.Linear(119, 64)  
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x):
        # 将输入展平为一维张量
        x = x.view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)

        return out


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def train(im_width, im_height, depth):


    feature_encoder = CNNEncoder()
    Prototype_network = PrototypeGenerator()

    feature_encoder.apply(weights_init)
    Prototype_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    Prototype_network.cuda(GPU)
    
    feature_encoder.train()
    Prototype_network.train()

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(optimizer=feature_encoder_optim, step_size=5000, gamma=0.5)
    # 每过step_size次,更新一次学习率;每经过100000次，学习率折半
    Prototype_network_optim = torch.optim.Adam(Prototype_network.parameters(), lr=LEARNING_RATE)
    Prototype_network_scheduler = StepLR(Prototype_network_optim, step_size=5000, gamma=0.5)



    # 训练数据集
    f = h5py.File('./data/source_data/meta_train_11000_78400.h5')
    train_dataset = f['data'][:]
    f.close()
    train_dataset = train_dataset.reshape(-1, n_examples, im_width, im_height, depth) 
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))[:, :, np.newaxis, :, :, :]
    print("train_dataset.shape", train_dataset.shape) 
    n_train_classes = train_dataset.shape[0]


    accuracy_ = []
    loss_ = []
    a = time.time()
    for episode in range(EPISODE):



        # start:每一个episode的采样过程##########################################################################################
        epi_classes = np.random.permutation(n_train_classes)[:n_way] 
        support = np.zeros([n_way, n_shot, 1, depth, im_height, im_width], dtype=np.float32) 
        query = np.zeros([n_way, n_query,  1, depth, im_height, im_width], dtype=np.float32) 


        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query] # 支撑集合
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]

        support = support.reshape(n_way * n_shot, 1, depth, im_height, im_width)
        query = query.reshape(n_way * n_query, 1, depth, im_height, im_width)
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8).reshape(-1)


        support_tensor = torch.from_numpy(support)
        query_tensor = torch.from_numpy(query)
        label_tensor = torch.LongTensor(labels)


        one_hot_labels = np.zeros((len(epi_classes), 55))

        # 填充独热编码，前16位有效，后4位补零
        for i, cls in enumerate(epi_classes):
           one_hot_labels[i][cls] = 1

        # 现在 one_hot_labels 包含了根据 epi_classes 动态生成的独热编码
        one_hot_labels_tensor = torch.from_numpy(one_hot_labels).float()
        one_hot_labels_tensor = one_hot_labels_tensor.cuda(GPU)


        # end:每一个episode的采样过程##########################################################################################
        sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  
        sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-4], list(sample_features.size())[-3],
                                               list(sample_features.size())[-2], list(sample_features.size())[
                                                   -1]) 
        sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均
        batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))

        ################################################################################################################
        sample_features = sample_features.view(n_way, list(sample_features.size())[1]*list(sample_features.size())[2],
                                               list(sample_features.size())[-2], list(sample_features.size())[-1])
        batch_features = batch_features.view(n_way*n_query, list(batch_features.size())[1] * list(batch_features.size())[2],
                                               list(batch_features.size())[-2], list(batch_features.size())[-1])
        ################################################################################################################
        
        sample_features = sample_features.squeeze(-1).squeeze(-1)
        sample_features_with_labels = torch.cat([sample_features, one_hot_labels_tensor], dim=1)

        prototypes = Prototype_network(sample_features_with_labels)


        batch_features = batch_features.squeeze(-1).squeeze(-1)
        crossEntropy = nn.CrossEntropyLoss().cuda(GPU)
        logits = euclidean_metric(batch_features, prototypes)
        loss = crossEntropy(logits, label_tensor.cuda().long())


        # training
        # 把模型中参数的梯度设为0
        feature_encoder.zero_grad()
        Prototype_network.zero_grad()
        loss.backward()

        # 进行单次优化，参数更新
        feature_encoder_optim.step()
        Prototype_network_optim.step()

        feature_encoder_scheduler.step()
        Prototype_network_scheduler.step()


        if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss",loss)
            #################调试#################
            _, predict_label = torch.max(logits, 1)
            predict_label = predict_label.cpu().numpy().tolist()
            rewards = [1 if predict_label[j] == labels[j] else 0 for j in range(labels.shape[0])]
            total_rewards = np.sum(rewards)

            accuracy = total_rewards*100.0 / labels.shape[0]
            print("accuracy:", accuracy)
            accuracy_.append(accuracy)
            loss_.append(loss.item())
    print('time = ',time.time()-a)



    torch.save(feature_encoder.state_dict(),str('./model/meta_training_feature_encoder_' +str(n_way) + 'way_' + str(n_shot) + 'shot.pkl'))
    torch.save(Prototype_network.state_dict(),str('./model/meta_training_network_' +str(n_way) + 'way_' + str(n_shot) + 'shot.pkl'))


    f = open('./result/meta_training_loss_' +str(n_way) + 'way_' + str(n_shot) + 'shot.txt', 'w')
    for i in range(np.array(loss_).shape[0]):
        f.write(str(loss_[i]) + '\n')
    f = open('./result/meta_training_accuracy_' +str(n_way) + 'way_' + str(n_shot) + 'shot.txt', 'w')
    for i in range(np.array(accuracy_).shape[0]):
        f.write(str(accuracy_[i]) + '\n')



if __name__ == '__main__':
    train(im_width, im_height, depth)