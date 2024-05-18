import os
from cat_model import cat_model
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from urllib.request import urlopen
from train import train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from image_dataset import image_dataset

if __name__ == '__main__':
    inputSize = 1200
    dataTransformsTrain = transforms.Compose([
        #transforms.RandomResizedCrop(inputSize),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataTransformsValid = transforms.Compose([
        #transforms.Resize(inputSize),
        #transforms.CenterCrop(inputSize),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainDatasets = image_dataset("./dataset_cnn/train.csv", "", dataTransformsTrain)
    validDatasets = image_dataset("./dataset_cnn/valid.csv", "", dataTransformsValid)
    dataloadersTrain = torch.utils.data.DataLoader(trainDatasets,
                                                   batch_size=4,
                                                   shuffle=True,
                                                   num_workers=8)
    dataloadersValid = torch.utils.data.DataLoader(validDatasets,
                                                   batch_size=4,
                                                   shuffle=False)

    # load model
    model = cat_model(in_channels=1, features= 8, num_classes=2).to(device)

    # set optimization function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    weights = torch.tensor([0.16, 0.84], device=device)  # 非热点权重为1，热点权重为10
    criterion = torch.nn.CrossEntropyLoss()

    # training
    model_ft = train(model, dataloadersTrain, dataloadersValid, optimizer, criterion, 20)

