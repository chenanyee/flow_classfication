import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from ae_model import ae_model
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ae_model().to(device)
model.load_state_dict(torch.load('./model/epoch_1_0.0346.pth'))
model.eval()

criterion = nn.MSELoss(reduction='mean')

dataTransforms = transforms.Compose([
    transforms.ToTensor(),
])

loss_dist = []

anom = pd.read_csv('./dataset/test.csv')

with torch.no_grad():
    for i in range(len(anom)):
        # 读取图像并进行预处理
        img_path = anom.iloc[i]['path']
        img = Image.open(img_path).convert('L')
        img = dataTransforms(img).float().to(device)
        img = img.unsqueeze(0)  # 添加批次维度
        
        # 前向传播
        sample = model(img)
        loss = criterion(img, sample)
        
        # 存储损失值
        loss_dist.append(loss.item())

# 可视化损失值
loss_sc = [(i, i) for i in loss_dist]
plt.scatter(*zip(*loss_sc))
plt.axvline(0.3, 0.0, 1)
plt.xlabel('Loss Value')
plt.ylabel('Loss Value')
plt.title('Loss Distribution')
plt.savefig('./model/test_loss_plot.png')
print("评估完成。")

lower_threshold = 0.06
upper_threshold = 0.25
plt.figure(figsize=(12,6))
plt.title('Loss Distribution')
sns.distplot(loss_dist,bins=100,kde=True, color='blue')
plt.axvline(upper_threshold, 0.0, 10, color='r')
plt.axvline(lower_threshold, 0.0, 10, color='b')
plt.savefig('./model/test_loss_disstru.png')

# 创建一个空的 DataFrame ddf，用来存储检测到的异常样本
ddf = pd.DataFrame(columns=anom.columns)

tp = 0  # True Positive: 真实为正，且被模型正确检测为正
fp = 0  # False Positive: 真实为负，但被模型错误检测为正
tn = 0  # True Negative: 真实为负，且被模型正确检测为负
fn = 0  # False Negative: 真实为正，但被模型错误检测为负
total_anom = 0 
# 遍历所有的损失值
for i in range(len(loss_dist)):
    label = anom.iloc[i]['label']
    # 将标签 'N' 和 'H' 转换为 0 和 1
    if label == 'N':
        numeric_label = 0
    elif label == 'H':
        numeric_label = 1
    else:
        raise ValueError(f"Unexpected label value: {label}")
    
    # 统计总的异常样本数
    total_anom += numeric_label

    # 判断当前损失值是否超过上界阈值（upper_threshold）
    if loss_dist[i] >= upper_threshold:
        
        # 如果当前样本的真实标签为 1，增加 TP 计数器
        if numeric_label == 1:
            tp += 1
        else:
            # 否则，增加 FP 计数器
            fp += 1
    else:
        if numeric_label == 1:
            fn += 1
        else:
            # 否则，增加 TN 计数器
            tn += 1

# 输出结果
print('[TP] {}\t[FP] {}\t[MISSED] {}'.format(tp, fp, total_anom-tp))
print('[TN] {}\t[FN] {}'.format(tn, fn))
