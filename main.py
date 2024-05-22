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
import random
from urllib.request import urlopen
from train import train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from image_dataset import image_dataset

if __name__ == '__main__':
    seed =0;
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    inputSize = 1200
    dataTransformsTrain = transforms.Compose([
        #transforms.RandomResizedCrop(inputSize),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataTransformsValid = transforms.Compose([
        #transforms.Resize(inputSize),
        #transforms.CenterCrop(inputSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainDatasets = image_dataset("./dataset/train.csv", "", dataTransformsTrain)
    validDatasets = image_dataset("./dataset/test.csv", "", dataTransformsValid)
    dataloadersTrain = torch.utils.data.DataLoader(trainDatasets,
                                                   batch_size=4,
                                                   shuffle=True,
                                                   num_workers=8)
    dataloadersValid = torch.utils.data.DataLoader(validDatasets,
                                                   batch_size=4,
                                                   shuffle=False,
                                                   num_workers=8)

    # load model
    model = cat_model(in_channels=1, num_classes=2).to(device)

    # set optimization function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    weights = torch.tensor([0.1, 1.1], device=device)  # 非热点权重为1，热点权重为10
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))

    # training
    model_ft = train(model, dataloadersTrain, dataloadersValid, optimizer, criterion, 15)

