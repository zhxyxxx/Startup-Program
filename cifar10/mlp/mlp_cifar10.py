import torch
from torchvision import datasets as dsets
from torchvision import transforms
from torch import nn
from torch.functional import F
from torch import optim
import matplotlib.pyplot as plt

train_Data = dsets.CIFAR10(
    root='../data_cifar10',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = dsets.CIFAR10(
    root='../data_cifar10',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_data, valid_data = torch.utils.data.random_split(train_Data, [40000, 10000])

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

#MLP(hidden3, 3072->1024->1024->512->10, relu)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MLP().to(device)

#loss:CrossEntropy
criterion = nn.CrossEntropyLoss()
#optimizer:SGD, learning rate:0.001
optimizer = optim.SGD(net.parameters(), lr=0.002)

#training

#100epochs
num_epochs = 100

train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

for epoch in range(num_epochs):
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

    #train
    net.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.view(-1, 32*32*3).to(device), labels.to(device)
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
            inputs, labels = inputs.view(-1, 32*32*3).to(device), labels.to(device)
            outputs = net(inputs)  #output
            loss = criterion(outputs, labels)  #loss
            val_loss += loss.item()
            acc = (outputs.max(1)[1] == labels).sum()
            val_acc += acc.item()
    avg_val_loss = val_loss / len(valid_loader.dataset)
    avg_val_acc = val_acc / len(valid_loader.dataset)

    print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
                   .format(epoch+1, num_epochs, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

    #data for plot
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

test_acc = 0

#test
net.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.view(-1, 32*32*3).to(device), labels.to(device)
        outputs = net(inputs)
        acc = (outputs.max(1)[1] == labels).sum()
        test_acc += acc.item()
    print('test_accuracy: {} %'.format(100 * test_acc / len(test_loader.dataset)))


torch.save(net.state_dict(), 'mlp_cifar10.ckpt')  #save model

#plot
fig_loss = plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()
fig_loss.savefig("mlp_cifar10_loss.png")

fig_acc = plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()
fig_acc.savefig("mlp_cifar10_acc.png")
