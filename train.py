import torch
import os
import time
from datetime import timedelta
from collections import defaultdict
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate(net, validLoader, criterion):
    net.eval()
    total_loss = 0
    count = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in validLoader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item()

    average_loss  = running_loss / len(validLoader.dataset)
    print(f"Validation Loss: {average_loss:.5f}")
    return average_loss

def train(net, trainLoader, validLoader, optimizer, criterion, epochs):
    net.to(device)
    metrics = defaultdict(list)
    start = time.time()
    best_loss = float('inf')
    best_model = None
    for i in range(epochs):
        net.train()
        epoch_start = time.time()
        running_loss = 0.0
        for data in trainLoader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainLoader.dataset)
        metrics['train_loss'].append(epoch_loss)
        ep_end = time.time()
        print('-----------------------------------------------')
        print(f'[EPOCH] {i + 1}/{epochs}\n[LOSS] {epoch_loss:.5f}')
        print(f'Epoch Complete in {ep_end- epoch_start:.5f} seconds')

        if i % 4 == 0 or i == epochs - 1:  # Evaluate every 3 epochs or the last epoch
            tmp_loss= validate(net, validLoader, criterion)
            if tmp_loss < best_loss:
                best_loss = tmp_loss
                best_model = net
                model_path = f"./model/epoch_{i + 1}_{best_loss:.4f}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(best_model.state_dict(), model_path)

    end = time.time()
    print('-----------------------------------------------')
    print(f'[System Complete: {timedelta(seconds=end - start)}]')
    
    _, ax = plt.subplots(1,1,figsize=(15,10))
    ax.set_title('Loss')
    ax.plot(metrics['train_loss'])
    plt.savefig('./model/train_loss_plot.png')
    return best_model
