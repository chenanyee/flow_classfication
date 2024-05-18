import torch
import os
import pandas as pd
import torchvision.transforms as transforms
from cat_model import cat_model
from image_dataset import image_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
dataTransforms = transforms.Compose([
    transforms.ToTensor(),
])

# 加载测试数据集
test_csv_path = "./dataset_cnn/test.csv"


testDatasets = image_dataset(test_csv_path, "", dataTransforms)
testLoader = DataLoader(testDatasets, batch_size=4, shuffle=False, num_workers=8)

def test(net, testLoader, criterion):
    net.eval()
    totalLoss = 0
    accuracy = 0
    count = 0
    with torch.no_grad():
        for x, label in testLoader:
            x = x.to(device)
            label = label.to(device, dtype=torch.float)
            output = net(x).squeeze()
            loss = criterion(output, label)
            predicted = (output > 0.5).float()
            #_, predicted = torch.max(output.data, 1)
            count += label.size(0)
            accuracy += (predicted == label).sum().item()
            totalLoss += loss.item() * label.size(0)
    print(f"Num: {count:.8f}")    
    print(f"Test Loss: {totalLoss / count:.8f}")
    print(f"Test Accuracy: {accuracy / count * 100:.2f}%")
    return accuracy / count

# 加载模型
model = cat_model().to(device)
model.load_state_dict(torch.load('./model/epoch_2_0.34.pth'))

criterion = torch.nn.BCEWithLogitsLoss()

# 执行测试
test_accuracy = test(model, testLoader, criterion)
