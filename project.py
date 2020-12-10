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

def train(trainloader, net, epochs):
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    transform = transforms.Compose([transforms.RandomCrop(32), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainloader = torch.utils.data.DataLoader(A.AbstractsDataset("data-1", transform), batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(A.AbstractsDataset("test-data-1", transform), batch_size=4, shuffle=True, num_workers=2)

    net = Tutorial.TutorialNet()

    train(trainloader, net, 1)


