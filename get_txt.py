import os

air_stft_root = r"E:\研究生\研二上\音色\带宽扩展\worldpython\air_npy\test/"
bone_stft_root = r"E:\研究生\研二上\音色\带宽扩展\worldpython\bone_npy\test/"
#mfcc_root=r'D:\dataset\bone_speech\align_signal\mfcc\bone_npy\test/'

files = os.listdir(bone_stft_root)
#分别写入train.txt, test.txt
with open('test.txt', 'w') as f1:
    for file in files:
        file_path = os.path.join(bone_stft_root,file)
        if file_path.endswith('.npy'):
            print(file_path)
            num=file.split('bone')[1]
            air_path=air_stft_root+'air'+str(num)
            print(air_path)
            #mfcc_path=mfcc_root+'bone'+str(num)
            f1.write('%s %s\n'%(file_path,air_path))
print('成功！')
