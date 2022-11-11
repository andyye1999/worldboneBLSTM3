import librosa
from scipy.io import wavfile
import numpy as np

list_speech=[4,13,20,26,37,43,49,58,65,74,84,91,100,108,
      121,131,145,154,165,176,188,200,207,212,224,231,239,
      248,261,270,277,284,292,302,308,316,325,333,345,356,
      362,371,378,388,401,410,418,427,435,441,448,456,467,476,
      484,490,502,509,518,526,533,541,551,559,565,573,584,592,
      604,611,618,632,642,652,662,671]

audio_path=r'E:\code_bone_conduct\项目3-2018年均衡器改进\实现代码3\语音数据\对齐语音\bone_new.wav'
path=r'D:\dataset\bone_speech\align_signal\bone_train/'
wav, sr_ret = librosa.load(audio_path, sr=None)
n=len(list_speech)
for i in range(n-1):
    a=int(list_speech[i]*8000)
    b=int(list_speech[i+1]*8000)
    print(a)
    print(b)
    wav_new=wav[a:b]
    signal_new=wav_new*32767
    ################################################
    name_path=path+'bone'+str(i)+'.wav'
    wavfile.write(name_path, 8000, signal_new.astype(np.int16))

wav_n=wav[b:]
signal_new=wav_n*32767
################################################
name_path=path+'bone'+str(n-1)+'.wav'
wavfile.write(name_path, 8000, signal_new.astype(np.int16))