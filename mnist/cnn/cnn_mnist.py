import torch
from torchvision import datasets as dsets
from torchvision import transforms
from torch import nn
from torch.functional import F
from torch import optim
import matplotlib.pyplot as plt
import time

train_Data = dsets.MNIST(
    root='../data_mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = dsets.MNIST(
    root='../data_mnist',
    train=False,
    transform=transforms.ToTensor(),
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


#CNN(Conv1, Affine1)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  #32 filters 5*5
        self.pool1 = nn.MaxPool2d(2)      #2*2
        self.fc1 = nn.Linear(12 * 12 * 32, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, 12*12*32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = CNN().to(device)

#loss:CrossEntropy
criterion = nn.CrossEntropyLoss()
#optimizer:SGD, learning rate:0.01
optimizer = optim.SGD(net.parameters(), lr=0.01)

#training

#50epochs
num_epochs = 50

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
print('elapsed time per epoch: {:.2f}'.format(timeperepoch))

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

torch.save(net.state_dict(), 'cnn_mnist.ckpt')  #save model

#plot
fig_loss = plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()
fig_loss.savefig("cnn_mnist_loss.png")

fig_acc = plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()
fig_acc.savefig("cnn_mnist_acc.png")
