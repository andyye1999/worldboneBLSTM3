"""
音频双通道分离
"""
import os
import sys
import numpy as np
from scipy.io import wavfile


def split_channel(wav_path, left_wav_path, right_wav_path):
    """
    通道分离
    :param wav_path: wav音频的路径
    :param left_wav_path: 左声道的wav音频路径
    :param right_wav_path: 右声道的wav音频路径
    :return None:
    """
    try:
        sample_rate, wav_data = wavfile.read(wav_path)
        print(sample_rate)
        left = []
        right = []
        for item in wav_data:
            left.append(item[0])
            right.append(item[1])
        wavfile.write(left_wav_path, sample_rate, np.array(left))
        wavfile.write(right_wav_path, sample_rate, np.array(right))
    except IOError as e:
        print('error is %s' % str(e))
    except:
        print('other error', sys.exc_info())


if __name__ == '__main__':
    file_dir = r'D:\dataset\bone_speech\骨传导语音数据录制\qinghua_danju\隋高扬'
    file_list = os.listdir(file_dir)
    for file in file_list:
        print(file)
        speech_path=os.path.join(file_dir,file)
        num=file.split('(')[1]
        left_speech=r'D:\dataset\bone_speech\骨传导语音数据录制\qinghua_danju\air_suigaoyang\air_suigaoyang('+num
        right_speech=r'D:\dataset\bone_speech\骨传导语音数据录制\qinghua_danju\bone_suigaoyang\bone_suigaoyang('+num
        split_channel(speech_path, left_speech,right_speech)
