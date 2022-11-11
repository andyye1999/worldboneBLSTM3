import os
import torch
import librosa
from scipy.io import wavfile
import numpy as np
from model import BiLSTM3

"CUDA加速"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nor_data(data,mu,sigma):
    # m,n=data.shape
    # mu = data.mean(axis=1).reshape(m,1)
    # sigma = data.std(axis=1).reshape(m,1)
    spec = (data- mu) / sigma
    return spec

def unnor_data(data,mu,sigma):
    ori_data=data*sigma+mu
    return ori_data

air_mu=np.load('E:\zhaoweiwei\code\清华语料实验\隋高扬语音模型\mu_air.npy')
air_sigma=np.load('E:\zhaoweiwei\code\清华语料实验\隋高扬语音模型\sigma_air.npy')

noise_mu=np.load('E:\zhaoweiwei\code\清华语料实验\隋高扬语音模型\mu_out.npy')
noise_sigma=np.load('E:\zhaoweiwei\code\清华语料实验\隋高扬语音模型\sigma_out.npy')

if __name__ == "__main__":

    model1 = BiLSTM3().to(device)
    state_dict1 = torch.load(r'E:\zhaoweiwei\code\清华语料实验\隋高扬语音模型\noise_snr=20\model_test.pth')
    model1.load_state_dict(state_dict1['model'])

    test_dir = r'E:\zhaoweiwei\code\清华语料实验\评估文件\out_train'
    file_list = os.listdir(test_dir)
    for file in file_list:
        file_path=os.path.join(test_dir,file)
        # 读取音频
        signal_b, fs_b = librosa.load(file_path, sr=None)
        #幅度归一化
        max_signal = max(abs(signal_b))
        x_test = signal_b * 1.0 / max_signal
        #stft
        x_spra = librosa.stft(x_test, n_fft=256, hop_length=64, win_length=256)
        # 求相位谱
        phase = np.angle(x_spra)
        # 求幅度谱
        mag = np.abs(x_spra)
        log_mag = np.log(mag**2+1e-6)
        nor_mag = nor_data(log_mag, noise_mu, noise_sigma)
        #确定输入大小
        m, n = x_spra.shape
        test_data =nor_mag.reshape(1,m,n).astype(np.float32)
        test_data=torch.from_numpy(test_data).to(device)
        m_bone=model1(test_data).reshape(m,n)
        m_bone_new= (m_bone.cpu()).detach().numpy()


        o_m_bone_new=unnor_data(m_bone_new,air_mu,air_sigma)
        e_m_bone_new=np.exp(o_m_bone_new/2)
        y =e_m_bone_new * np.exp(1j * phase)
        signal_new=librosa.istft(y, hop_length=64, win_length=256, window='hann', center=True)
        signal_new=signal_new*32767*max_signal

        name_path=file.split('.')[0]+'_test.wav'
        wavfile.write(name_path, 16000, signal_new.astype(np.int16))















