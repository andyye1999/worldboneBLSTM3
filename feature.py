import librosa
import os
import numpy as np
import math
import pyworld as pw


n_fft=256
win_length=256
hop_length=64

path_new=r'E:\研究生\研二上\音色\带宽扩展\清华语料实验 - out\bone_npy\train/'
path_name=r'E:\研究生\研二上\音色\带宽扩展\清华语料实验 - out\bone\train/'
path_list=os.listdir(path_name)
for file in path_list:
    audio_path=os.path.join(path_name,file)
    # wav, sr_ret = librosa.load(audio_path,sr=None)
    wav16k, fs_16k = librosa.load(audio_path, sr=16000,dtype=np.float64) #librosa load输出的waveform 是 float32，

    fs_8k = 8000
    wav_8k = librosa.resample(wav16k, fs_16k, fs_8k)

    _, _, ap = pw.wav2world(wav16k, fs_16k)
    f0, sp, _ = pw.wav2world(wav_8k, fs_8k)
    ap = ap[:, :sp.shape[1]].copy(order="C")

    # max_signal = max(abs(wav))
    # x_test = wav * 1.0 / max_signal

    # linear = librosa.stft(x_test, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    str1=file.split('.')[0]

    m,n=sp.shape
    num=math.ceil((n-15)/15)
    z=0
    for i in range(num):
        name = path_new+ str(str1)+'-'+str(i) + '.npy'
        # np.save(name,sp[:,z:(z+256)])
        np.save(name,sp[z:(z+15),:])
        z=z+15
    name_1 = path_new + str(str1) + '-' + str(num) + '.npy'
    # np.save(name_1, sp[:, (n-15):])
    np.save(name_1, sp[(n-15):n,:])

