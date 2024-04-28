import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


transform = transforms.ToTensor()
batch_size = 4
trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
             
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')


class DigitRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.linear(784,512)
        self.activation = nn.ReLu()
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.linear(512,10)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, images):
        x= self.linear1(images)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


model = DigitRecognition()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 5  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(images.shape[0],784)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')