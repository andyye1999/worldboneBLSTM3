'''
import librosa
from scipy.io import wavfile
import numpy as np

def nor_data(data):
    m,n=data.shape
    mu = data.mean(axis=0).reshape(1, n)
    sigma = data.std(axis=0).reshape(1, n)
    spec = (data- mu) / sigma
    return spec,mu,sigma

def unnor_data(data,mu,sigma):
    ori_data=data*sigma+mu
    return ori_data
#读取音频
signal_b,fs_b=librosa.load(r'E:\code_bone_conduct\项目3-2018年均衡器改进\实现代码3\ht_16000\ht_163.wav', sr=None)
# max_signal=max(abs(signal_b))
# x_test=signal_b*1.0/max_signal
x_test=signal_b
x_spra = librosa.stft(x_test, n_fft=256, hop_length=64, win_length=256)
# 求相位谱
phase = np.angle(x_spra)
#求幅度谱
mag = np.abs(x_spra)
mag=np.log(mag)
mag,mu,sigma=nor_data(mag)

o_m_bone_new=unnor_data(mag,mu,sigma)
e_m_bone_new=np.exp(o_m_bone_new)
y =e_m_bone_new * np.exp(1j * phase)
signal_new=librosa.istft(y, hop_length=64, win_length=256, window='hann', center=True)
# signal_new=signal_new*32767*max_signal
signal_new=signal_new*32767
wavfile.write("bone_163.wav", 16000, signal_new.astype(np.int16))'''

import os
import numpy as np

def Normalize(spec):
    # (Freq, Time)
    # mean-variance normalization for every spectrogram (not batch-wise)
    m,n=spec.shape
    mu = spec.mean(axis=1).reshape(m,1)
    sigma = spec.std(axis=1).reshape(m,1)
    spec = (spec - mu) / sigma
    return spec,mu,sigma

file_dir=r'E:\zhaoweiwei\code\align_speech\out_npy\train'
file_list=os.listdir(file_dir)
mu_list=[]
sigma_list=[]
for file in file_list:
    path_name=os.path.join(file_dir,file)
    x=np.load(path_name)
    #计算log后数值太大，+ 1e-6为数据精度处理
    x=np.log(np.abs(x**2)+ 1e-6)
    spec,mu,sigma=Normalize(x)
    mu_list.append(mu)
    sigma_list.append(sigma)
mu_list=np.array(mu_list)
sigma_list=np.array(sigma_list)
mean_mu=np.mean(mu_list,axis=0)
mean_sigma=np.mean(sigma_list,axis=0)
print(mean_mu)
np.save('mu_out.npy', mean_mu)
print(mean_sigma.shape)
np.save('sigma_out.npy', mean_sigma)












