import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(net, testLoader, criterion):
    net.eval()
    totalLoss = 0
    accuracy = 0
    count = 0
    for x, label in testLoader:
        x = x.to(device)
        label = label.to(device, dtype=torch.long)
        output = net(x)
        loss = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        count += len(x)
        accuracy += (predicted == label).sum().item()
        totalLoss += loss.item()*len(label)
    print("Test Loss: {}".format(totalLoss / count))
    print("Test Accuracy: {}".format(accuracy / count))
    return (accuracy / count)

def train(net, trainLoader, testLoader, optimizer, criterion, epochs):
    net.train()
    testAccuracy = 0
    bestModel = net
    for i in range(epochs):
        totalLoss = 0
        accuracy = 0
        count = 0
        for x, label in trainLoader:
            x = x.to(device)
            label = label.to(device, dtype=torch.long)
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, label)
            _, predicted = torch.max(output.data, 1)
            count += len(x)
            accuracy += (predicted == label).sum().item()
            totalLoss += loss.item()*len(label)
            loss.backward()
            optimizer.step()
        print("Train Loss: {}".format(totalLoss / count))
        print("Train Accuracy: {}".format(accuracy / count))
        if (i % 10 == 0):
            tmpAccuracy = test(net, testLoader, criterion)
            if (tmpAccuracy > testAccuracy):
                testAccuracy = tmpAccuracy
                bestModel = net
                epoch = i
    torch.save(bestModel, "model/epoch"+str(epoch)+"_"+str(testAccuracy)+".pth")
    return net