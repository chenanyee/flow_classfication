import torch
import torch.nn as nn
import numpy as np
import pandas as pd

model = ae_model().to(device)
model.load_state_dict(torch.load('./model/.pth'))
model.eval()

criterion = nn.MSELoss(reduction='mean')
loss_dist = []

anom = pd.read_csv('./dataset/anom.csv', index_col=[0])

with torch.no_grad():
    for i in range(len(anom)):
        data = torch.from_numpy(np.array(anom.iloc[i][1:])/255).float().to(device)
        data = data.unsqueeze(0)  # 添加批次维度
        sample = model(data)
        loss = criterion(data, sample)
        loss_dist.append(loss.item())

print(f'平均损失：{np.mean(loss_dist)}')