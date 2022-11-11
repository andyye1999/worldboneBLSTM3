import torch
import os
import torch.nn as nn
from dataset import bone_Dataset
from tqdm import tqdm
from model import BiLSTM3
import torch.optim as optim
from torch.utils.data import Dataset	# 注意是大写的Data, 不是data \ Date
dtype = torch.FloatTensor
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
# 设置汉字输出
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
'''
保存验证集中的最优模型
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH_NUM=1000
trainset = bone_Dataset('train.txt')
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

validset = bone_Dataset('test.txt')
validsetloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)

model = BiLSTM3().to(device)
criterion = nn.MSELoss(reduction='mean').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)

min_loss = 100000
for epoch_num in range(EPOCH_NUM):
    model.train()#训练部分
    running_loss = 0.0#训练集batch损失累计
    loop = tqdm(enumerate(trainsetloader), total=len(trainsetloader))
    for iter_num, (train_data,train_label) in loop: #tqdm(enumerate(trainsetloader)):#iter_num表示一个batch
        train_data = train_data.to(device)
        train_label = train_label.to(device)
        optimizer.zero_grad()
        scores = model(train_data)
        train_loss = criterion(scores, train_label.float())
        train_loss.backward()
        optimizer.step()
        #记录一个batch的损失
        running_loss += train_loss.item()
        loop.set_description(f'Epoch [{epoch_num}/{EPOCH_NUM}]')
        loop.set_postfix(loss=train_loss.data.item())
        if iter_num % 200 == 0: # 每200个batch  250 x 128 进行一次记录
            print(
                "epoch {} - iteration {}: average loss {:.12f}".format(epoch_num + 1, iter_num + 1, running_loss / 200))
            running_loss = 0.0
    if (epoch_num+1) % 100 == 0:
        print("save train model")
        train_model_name='model_train_'+str(epoch_num)+'.pth'
        torch.save({'model': model.state_dict()}, train_model_name)
    model.eval()  # 验证部分
    running_test_loss=0.0
    with torch.no_grad():
        testloop = tqdm(enumerate(validsetloader), total=len(validsetloader))
        for i, (test_data,test_label) in testloop: #tqdm(enumerate(validsetloader)):
            test_data = test_data.to(device)
            test_label = test_label.to(device)
            test_outputs =model(test_data)
            test_loss = criterion(test_outputs, test_label)
            running_test_loss += test_loss.item()
            loop.set_description(f'Epoch [{epoch_num}/{EPOCH_NUM}]')
            loop.set_postfix(loss=test_loss.data.item())
        print("epoch:{} step:{} train Loss:{} test Loss:{}".format(epoch_num + 1, i, train_loss.item(), test_loss.item()))
        if running_test_loss < min_loss:
            min_loss = running_test_loss
            print("save test model")
            torch.save({'model': model.state_dict()}, 'model_test.pth')
    # torch.save(model,"model/cnn_image_model_epoch_{}_step{}.pkl".format(epoch,i))

