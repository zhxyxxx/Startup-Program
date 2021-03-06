import torch
from torchvision import datasets as dsets
from torchvision import transforms
from torchvision import models
from torch import nn
from torch.functional import F
from torch import optim
import matplotlib.pyplot as plt
import sys
import time
from efficientnet_pytorch import EfficientNet

NET = ["Lenet", "Mobilenet", "VGG", "Resnet", "Efficientnet"]

args = sys.argv
if len(args) != 2:
    print("Argument missed")
elif args[1] in NET:
    network = NET.index(args[1])
else:
    print("Undefined Argument")

if network == 0:
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

train_Data = dsets.MNIST(
    root='../data_mnist',
    train=True,
    transform=transform,
    download=True
)

test_data = dsets.MNIST(
    root='../data_mnist',
    train=False,
    transform=transform,
    download=True
)

train_data, valid_data = torch.utils.data.random_split(train_Data, [48000, 12000])

#load data
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_data,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False,
    num_workers=2
)


#Lenet
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1*1*120)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if network == 0:
    net = Lenet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)
elif network == 1:
    net = models.mobilenet_v2(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004)
elif network == 2:
    net = models.vgg11(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004)
elif network == 3:
    net = models.resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004)
elif network == 4:
    net = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 10}).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004)
else:
    print("Error")

print(args[1])

#training
num_epochs = 20

train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
elapsed_time = 0

for epoch in range(num_epochs):
    start = time.time()
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

    #train
    net.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  #initialize
        outputs = net(inputs)  #output
        loss = criterion(outputs, labels)  #loss
        train_loss += loss.item()
        acc = (outputs.max(1)[1] == labels).sum()
        train_acc += acc.item()
        loss.backward()  #backward
        optimizer.step()  #update weight
    avg_train_loss = train_loss / len(train_loader.dataset)  #calculate average loss
    avg_train_acc = train_acc / len(train_loader.dataset)  #average accuracy

    #valid
    net.eval()
    with torch.no_grad():  #stop calculation of grad
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)  #output
            loss = criterion(outputs, labels)  #loss
            val_loss += loss.item()
            acc = (outputs.max(1)[1] == labels).sum()
            val_acc += acc.item()
    avg_val_loss = val_loss / len(valid_loader.dataset)
    avg_val_acc = val_acc / len(valid_loader.dataset)

    elapsed_time += time.time() - start
    print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
                   .format(epoch+1, num_epochs, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

    #data for plot
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

timeperepoch = elapsed_time / num_epochs
print('elapsed time per epoch: {:.4f}'.format(timeperepoch))

test_acc = 0

#test
net.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        acc = (outputs.max(1)[1] == labels).sum()
        test_acc += acc.item()
    print('test_accuracy: {} %'.format(100 * test_acc / len(test_loader.dataset)))


model_file = args[1] + "_mnist.ckpt"
torch.save(net.state_dict(), model_file)  #save model

#plot
fig_file = args[1] + "_mnist.png"
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))

ax1.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
ax1.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
ax1.legend()
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('Training and validation loss')
ax1.grid()

ax2.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
ax2.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('acc')
ax2.set_title('Training and validation accuracy')
ax2.grid()

fig.savefig(fig_file)
