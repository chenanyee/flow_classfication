import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate(net, validLoader, criterion):
    net.eval()
    totalLoss = 0
    accuracy = 0
    count = 0
    with torch.no_grad():
        for x, label in validLoader:
            x, label = x.to(device), label.to(device, dtype=torch.float)
            output = net(x).squeeze()
            loss = criterion(output, label)
            predicted = (output > 0.5).float()
            #_, predicted = torch.max(output.data, 1)
            count += label.size(0)
            accuracy += (predicted == label).sum().item()
            totalLoss += loss.item() * label.size(0)
    print(f"Validation Loss: {totalLoss / count:.8f}")
    print(f"Validation Accuracy: {accuracy / count * 100:.4f}%")
    return accuracy / count

def train(net, trainLoader, validLoader, optimizer, criterion, epochs):
    net.to(device)
    bestAccuracy = 0
    bestModel = None
    for i in range(epochs):
        net.train()
        totalLoss = 0
        accuracy = 0
        count = 0
        for x, label in trainLoader:
            x = x.to(device)
            label = label.to(device, dtype=torch.float)
            optimizer.zero_grad()
            #output = net(x)
            output = net(x).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            #_, predicted = torch.max(output.data, 1)
            predicted = (output > 0.5).float()
            count += label.size(0)
            accuracy += (predicted == label).sum().item()
            totalLoss += loss.item() * label.size(0)
        print(f"Epoch: {i + 1}/{epochs}")
        print(f"Train Loss: {totalLoss / count:.8f}")
        print(f"Train Accuracy: {accuracy / count * 100:.2f}%\n")

        if i == epochs - 1:  # Evaluate every 3 epochs or the last epoch
            tmpAccuracy = validate(net, validLoader, criterion)
            if tmpAccuracy > bestAccuracy:
                bestAccuracy = tmpAccuracy
                bestModel = net
                model_path = f"model/epoch_{i + 1}_{bestAccuracy:.2f}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(bestModel.state_dict(), model_path)
    return bestModel
