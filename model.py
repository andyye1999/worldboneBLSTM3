import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM3(nn.Module):
    def __init__(self):
        super(BiLSTM3, self).__init__()
        self.lstm1 = nn.LSTM(input_size=257, hidden_size=512, num_layers=3,dropout=0.02,bidirectional=True)
        self.fc = nn.Linear(512*2, 257)# fc
    def forward(self, X):
        batch_size, m, z = X.shape # batch,seq,inputsize
        #print(out.shape)
        input = X.permute(1,0,2)#指定维度新的位置 # (seq_length,batch_size,input_size)
        h1 = torch.randn(3*2, batch_size, 512).to(device)  # [num_layers(=3) * num_directions(=2), batch_size, n_hidden]
        c1 = torch.randn(3*2, batch_size, 512).to(device)

        outputs1, (_, _) = self.lstm1(input, (h1, c1)) # (seq_length,batch_size,num_directions*hidden_size)
        outputs = outputs1.permute(1,0,2)   # (batch_size,seq_length,num_directions*hidden_size)
        model=self.fc(outputs)
        # model=model.permute(0,2,1)
        return model


if __name__=="__main__":
    input = torch.randn(128,15,257)
    model=BiLSTM3().to(device)
    print(model)
    output=model(input.to(device))
    print(output.shape)
    print('cliqueNet parameters:', sum(param.numel() for param in model.parameters()))
    print('cliqueNet parameters1:',sum(param.numel() for param in model.parameters() if param.requires_grad))