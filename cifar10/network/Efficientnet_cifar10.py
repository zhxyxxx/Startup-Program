import torch
from torchvision import datasets as dsets
from torchvision import transforms
from torch import nn
from torch import optim
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import time

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# prepare data
train_Data = dsets.CIFAR10(
    root='../data_cifar10',
    train=True,
    transform=transform,
    download=False
)

test_data = dsets.CIFAR10(
    root='../data_cifar10',
    train=False,
    transform=transform,
    download=False
)

train_data, valid_data = torch.utils.data.random_split(train_Data, [40000, 10000])

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net0 = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': 10}).to(device)
net1 = EfficientNet.from_name('efficientnet-b1', override_params={'num_classes': 10}).to(device)
net2 = EfficientNet.from_name('efficientnet-b2', override_params={'num_classes': 10}).to(device)
net3 = EfficientNet.from_name('efficientnet-b3', override_params={'num_classes': 10}).to(device)
net4 = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 10}).to(device)
net5 = EfficientNet.from_name('efficientnet-b5', override_params={'num_classes': 10}).to(device)
net6 = EfficientNet.from_name('efficientnet-b6', override_params={'num_classes': 10}).to(device)
net7 = EfficientNet.from_name('efficientnet-b7', override_params={'num_classes': 10}).to(device)
NET = [net0, net1, net2, net3, net4, net5, net6, net7]


Train_loss_lists, Train_acc_lists, Val_loss_lists, Val_acc_lists = [], [], [], []
time_list, acc_list = [], []
for i, net in enumerate(NET):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005)

    print("EfficientNet-B{}".format(i))

    #training
    num_epochs = 1

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
        accuracy = 100 * test_acc / len(test_loader.dataset)
        print('test_accuracy: {} %'.format(accuracy))


    Train_loss_lists.append(train_loss_list)
    Train_acc_lists.append(train_acc_list)
    Val_loss_lists.append(val_loss_list)
    Val_acc_lists.append(val_acc_list)
    time_list.append(timeperepoch)
    acc_list.append(accuracy)


# plot
colorlist = ['red', 'blue', 'orange', 'green', 'gold', 'magenta', 'yellow', 'cyan']
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))

for i, (train_loss_list, val_loss_list, color) in enumerate(zip(Train_loss_lists, Val_loss_lists, colorlist)):
    tlabel = 'train_B{}'.format(i)
    vlabel = 'val_B{}'.format(i)
    ax1.plot(range(num_epochs), train_loss_list, color=color, linestyle='-', label=tlabel)
    ax1.plot(range(num_epochs), val_loss_list, color=color, linestyle='--', label=vlabel)
ax1.legend()
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_title('Training and validation loss')
ax1.grid()

for i, (train_acc_list, val_acc_list, color) in enumerate(zip(Train_acc_lists, Val_acc_lists, colorlist)):
    ax2.plot(range(num_epochs), train_acc_list, color=color, linestyle='-')
    ax2.plot(range(num_epochs), val_acc_list, color=color, linestyle='--')
ax2.set_xlabel('epoch')
ax2.set_ylabel('acc')
ax2.set_title('Training and validation accuracy')
ax2.grid()

fig.savefig('Efficientnet_cifar10.png')


fig2 = plt.figure()

for i, (time, acc) in enumerate(zip(time_list, acc_list)):
    plt.plot(time, acc, 'o')
    plt.annotate(i, xy=(time, acc))
plt.xlabel('time')
plt.ylabel('accuracy')
plt.title('Time and Accuracy')
plt.grid()
fig2.savefig("Efficientnet_cifar10_time.png")
