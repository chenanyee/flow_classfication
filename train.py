import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate(net, validLoader, criterion):
    net.eval()
    totalLoss = 0
    total_accuracy = 0
    correct_1 = 0
    false_positive = 0
    count = 0
    total_1 = 0
    with torch.no_grad():
        for x, label in validLoader:
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
    print(f"Test Loss: {totalLoss / count:.8f}")
    print(f"Test Toal_Accuracy: {total_accuracy / count * 100:.4f}%")
    print(f"Test False Alarm: {false_positive}")
    print(f"Validation Hostpot_Accuracy: {correct_1/ total_1 * 100:.2f}%\n")
    
    return correct_1/ total_1 + total_accuracy / count

def train(net, trainLoader, validLoader, optimizer, criterion, epochs):
    net.to(device)
    bestAccuracy = 0
    bestModel = None
    for i in range(epochs):
        net.train()
        running_loss = 0
        train_total_accuracy = 0
        correct_1 = 0
        count = 0
        total_1 = 0
        for x, label in trainLoader:
            x = x.to(device)
            label = label.to(device)
            
            output = net(x)
            optimizer.zero_grad()
            #output = net(x).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            
            #predicted = (output > 0.5).float()
            count += label.size(0)
            total_1 += (label == 1).sum().item()
            train_total_accuracy += (predicted == label).sum().item()
            correct_1 += ((predicted == 1) & (label == 1)).sum().item()
            running_loss += loss.item()
            #totalLoss += loss.item() * label.size(0)
        print(f"Epoch: {i + 1}/{epochs}")
        print(f"Train Loss: {running_loss / count:.8f}")
        print(f"Train Toal_Accuracy: {train_total_accuracy / count * 100:.2f}%")
        print(f"Train Hostpot_Accuracy: {correct_1/ total_1 * 100:.2f}%\n")

        if i == epochs - 1 or i % 3 == 0:  # Evaluate every 3 epochs or the last epoch
            tmpAccuracy = validate(net, validLoader, criterion)
            if tmpAccuracy > bestAccuracy:
                bestAccuracy = tmpAccuracy
                bestModel = net
                model_path = f"model/cnn_epoch_{i + 1}_{bestAccuracy:.2f}.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(bestModel.state_dict(), model_path)
    return bestModel
