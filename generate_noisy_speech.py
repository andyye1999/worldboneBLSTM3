import numpy as np
import librosa
import os
from scipy.io import wavfile
def add_niose(data,snr):
    N = data.size
    Ps = np.sum(data ** 2 / N)
    # Signal power, in dB
    Psdb = 10 * np.log10(Ps)
    # Noise level necessary
    Pn = Psdb - snr
    # Noise vector (or matrix)
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, data.shape)
    return data+n

path_noise=r'E:\zhaoweiwei\code\align_speech\noise\snr=20\test/'
path_bone=r'E:\zhaoweiwei\code\align_speech\bone\test'
path_list=os.listdir(path_bone)
for file in path_list:
    audio_path=os.path.join(path_bone,file)
    wav, sr_ret = librosa.load(audio_path,sr=None)
    audio_niose=add_niose(wav,snr=20)
    signal_new=audio_niose*32767
    name_path=path_noise+'niose'+file.split('bone')[1]
    wavfile.write(name_path, 16000, signal_new.astype(np.int16))