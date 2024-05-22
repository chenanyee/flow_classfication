import torch
import os
import pandas as pd
import torchvision.transforms as transforms
from cat_model import cat_model
from image_dataset import image_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix using seaborn heatmap.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("./model/confusion_res.png")

def test(net, testLoader, criterion):
    net.eval()
    totalLoss = 0
    total_accuracy = 0
    correct_1 = 0
    false_positive = 0
    count = 0
    total_1 = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for x, label in testLoader:
            x, label = x.to(device), label.to(device)
            
            output = net(x)
            loss = criterion(output, label)
            _, predicted = torch.max(output.data, 1)
            
            count += label.size(0)
            total_accuracy += (predicted == label).sum().item()
            correct_1 += ((predicted == 1) & (label == 1)).sum().item()
            total_1 += (label == 1).sum().item()
            false_positive += ((predicted == 1) & (label == 0)).sum().item()
            totalLoss += loss.item()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(f"Test Loss: {totalLoss / count:.8f}")
    print(f"Test Total Accuracy: {total_accuracy / count * 100:.4f}%")
    print(f"Test False Alarm: {false_positive}")
    print(f"Validation Hotspot Accuracy: {correct_1 / total_1 * 100:.2f}%\n")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    class_names = ["Non-Hotspot", "Hotspot"]  # Adjust based on your class names
    plot_confusion_matrix(cm, class_names)

    return correct_1 / count
    

# 加载模型
model = cat_model().to(device)
model.load_state_dict(torch.load('./model/epoch_5_0.99.pth'))

# 数据预处理
dataTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载测试数据集
test_csv_path = "./dataset/test.csv"


testDatasets = image_dataset(test_csv_path, "", dataTransforms)
testLoader = DataLoader(testDatasets, batch_size=4, shuffle=False, num_workers=8)

criterion = torch.nn.CrossEntropyLoss()

# 执行测试
test_accuracy = test(model, testLoader, criterion)
