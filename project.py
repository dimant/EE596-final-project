import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import crnn
import abstracts_dataset as A
import tutorial_net as Tutorial
from torch.autograd import Variable

# http://www.ironicsans.com/helvarialquiz/

fonts = [
      "Arial", 
      "Times New Roman", 
      "Courier New", 
      "Calibri", 
      "Candara", 
      "Georgia", 
      "Corbel",
      "Helvetica",
      "Comic Sans MS",
      "Garamond"]

def test_net(data, net):
    global fonts
    correct = 0
    total = 0
    with torch.no_grad():
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        keys = labels.unique()
        class_accuracy = {}
        class_counts = {}

        for key in keys:
            class_accuracy[key.item()] = 0
            class_counts[key.item()] = 0
            for l in labels:
                if l.item() == key.item():
                    class_counts[key.item()] += 1


        for i in range(len(labels)):
            l = labels[i].item()
            if predicted[i] == l:
                class_accuracy[l] += 1

        for key in keys:
            class_accuracy[key.item()] /= class_counts[key.item()]

    return 100.0 * correct / total, class_accuracy

def train(trainloader, testloader, net, epochs):
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters())

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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
                accuracy, class_accuracy = test_net(next(iter(testloader)), net)
                print('[%d, %5d] loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 100, accuracy))
                castr = ""
                for i in range(len(fonts)):
                    castr += fonts[i] + " {0:.3g}".format(class_accuracy[i]* 100) + "% "
                print("class accuracy: " + castr)
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    transform = transforms.Compose([
        transforms.RandomCrop(32), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # FONTS = [
    #   "Arial", 
    #   "Times New Roman", 
    #   "Courier New", 
    #   "Calibri", 
    #   "Candara", 
    #   "Consolas", 
    #   "Georgia", 
    #   "Corbel", 
    #   "Arial Black"]

    print("Loading training data...")
    data1 = A.AbstractsDataset("data-1", transform, fonts, 100000)
    trainloader = torch.utils.data.DataLoader(data1, batch_size=100, shuffle=True, num_workers=2)
    print("loaded %d images" % (data1.__len__()))

    print("Loading test data...")
    traindata1 = A.AbstractsDataset("test-data-1", transform, fonts, 1000)
    testloader = torch.utils.data.DataLoader(traindata1, batch_size=700, shuffle=False, num_workers=2)
    print("loaded %d images" % (traindata1.__len__()))

    print("Training network...")

    # net = Tutorial.TutorialNet(num_classes=len(fonts))

    # net = torchvision.models.resnet18(pretrained=False, progress=False)
    # net.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # net.fc = nn.Linear(in_features=512, out_features=len(fonts), bias=True)

    # image height, number of channels, number of classes, size of the lstm hidden state
    net = crnn.CRNN(fonts)

    train(trainloader, testloader, net, 3)

