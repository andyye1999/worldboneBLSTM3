from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np

"""Normalizes voice spectrogram (mean-varience)"""
def normalize(spec,mu,sigma):
    # (Freq, Time)
    # mean-variance normalization for every spectrogram (not batch-wise)
    spec = (spec - mu) / sigma
    return spec

def ToTensor(spec):
    #F, T = spec.shape
    # now specs are of size (Freq, Time) and 2D but has to be 3D (channel dim)
    #spec = spec.reshape(1, F, T)
    # make the ndarray to be of a proper type (was float64)
    spec = spec.astype(np.float32)
    return torch.from_numpy(spec)

# air_mu=np.load('E:\zhaoweiwei\code\清华语料实验 - out\隋高扬语音模型\mu_air.npy')
# air_sigma=np.load('E:\zhaoweiwei\code\清华语料实验 - out\隋高扬语音模型\sigma_air.npy')
#
# noise_mu=np.load('E:\zhaoweiwei\code\清华语料实验 - out\隋高扬语音模型\mu_out.npy')
# noise_sigma=np.load('E:\zhaoweiwei\code\清华语料实验 - out\隋高扬语音模型\sigma_out.npy')

# bone_mu=np.load('E:\zhaoweiwei\code\清华语料实验\隋高扬语音模型\mu_bone.npy')
# bone_sigma=np.load('E:\zhaoweiwei\code\清华语料实验\隋高扬语音模型\sigma_bone.npy')
# bone_mu=np.load('E:\code_bone_conduct\项目3-2018年均衡器改进\实现代码3\隋高扬语音模型\mean_mu_bone.npy')
# bone_sigma=np.load('E:\code_bone_conduct\项目3-2018年均衡器改进\实现代码3\隋高扬语音模型\mean_sigma_bone.npy')

class bone_Dataset(Dataset):  # 继承Dataset
    def __init__(self, txt):
        with open(txt, 'r') as fh:
            bone_path = []
            air_path = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split()  # 以空格为分隔符 将字符串分成a
                bone_path.append(words[0])
                air_path.append(words[1])
        self.bone_list = bone_path
        self.air_list = air_path
    def __len__(self):
        return len(self.bone_list)  # 返回数据的数量
    def __getitem__(self, idx):
        #n_frames=256
        # 得到训练数据
        train_data_path = self.bone_list[idx]
        x_b = np.load(train_data_path)
        label_data_path = self.air_list[idx]
        x_a = np.load(label_data_path)
        # train_data = np.log(np.abs(x_b ** 2) + 1e-6)
        # label_data = np.log(np.abs(x_a ** 2) + 1e-6)
        # train_data = normalize(train_data, noise_mu, noise_sigma)
        train_data = ToTensor(x_b)
        # label_data = normalize(label_data, air_mu, air_sigma)
        label_data = ToTensor(x_a)
        return train_data, label_data  # 返回图片, 标签元组