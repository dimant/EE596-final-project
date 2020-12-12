import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import abstracts_dataset as A
import tutorial_net as Tutorial

# http://www.ironicsans.com/helvarialquiz/

def test_net(data, net):
    correct = 0
    total = 0
    with torch.no_grad():
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 - 100.0 * correct / total

def train(trainloader, testloader, net, epochs):
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters())

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                accuracy = test_net(next(iter(testloader)), net)
                print('[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 100, accuracy))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    transform = transforms.Compose([transforms.RandomCrop(32), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # FONTS = [
    #   "Arial", 
    #   "Times New Roman", 
    #   "Comic Sans", 
    #   "Courier New", 
    #   "Calibri", 
    #   "Candara", 
    #   "Consolas", 
    #   "Georgia", 
    #   "Corbel", 
    #   "Arial Black"]
    fonts = ["Arial", "Times New Roman", "Comic Sans", "Courier New", "Calibri", ]

    print("Loading training data...")
    trainloader = torch.utils.data.DataLoader(A.AbstractsDataset("data-1", transform, fonts), batch_size=10, shuffle=True, num_workers=2)

    print("Loading test data...")
    testloader = torch.utils.data.DataLoader(A.AbstractsDataset("test-data-1", transform, fonts), batch_size=200, shuffle=True, num_workers=2)

    print("Training network...")
    # net = Tutorial.TutorialNet()

    # load the model
    net = torchvision.models.resnet18(pretrained=False)

    # replace the last layer
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, len(fonts))

    train(trainloader, testloader, net, 1)


