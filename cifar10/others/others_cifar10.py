import torch
from torchvision import datasets as dsets
from torchvision import transforms
from torchvision import models
from torch import nn
from torch.functional import F
from torch import optim
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import argparse
import time
import logging
import sys

NET = ["Lenet", "Lenet_r", "Mobilenet", "VGG", "VGG_bn", "VGG_bn_nw", "Resnet50", "Resnet101", "EfficientnetB0", "EfficientnetB1", "EfficientnetB2", "EfficientnetB3"]

# command line argument
parser = argparse.ArgumentParser()
parser.add_argument('network', choices=NET, help='choose network')
parser.add_argument('-l', '--lr', type=float, default=0.005, help='set learning rate for optimizer')
parser.add_argument('-e', '--epoch', type=int, default=50, help='set number of epochs')
parser.add_argument('-b', '--batch', type=int, default=64, help='set mini batch size')
parser.add_argument('--m', type=float, default=0, help='set momentum for optimizer')
parser.add_argument('--wd', type=float, default=0, help='set weight decay for optimizer')
parser.add_argument('--test', action='store_true', help='test mode, with no log, model and graph output')
parser.add_argument('--load', action='store_true', help='load your model (the file name should be *_cifar10.ckpt)')
parser.add_argument('--use32', action='store_true', help='train with 32*32 graph size without resizing')
parser.add_argument('--para', action='store_true', help='use multi GPU')
parser.add_argument('--noise', action='store_true', help='add noise to grad')
parser.add_argument('--lrdecay', type=int, choices=range(-1, 3), default=-1, help='use learning rate decay')
parser.add_argument('--dr', type=float, default=0.1, help='set decay rate for learning rate')
parser.add_argument('--smooth', action='store_true', help='use smoothout')
args = parser.parse_args()

model_file = args.network + "_cifar10.ckpt"
fig_file = args.network + "_cifar10.png"
network = NET.index(args.network)
if network <= 1 or args.use32:
    transform = transforms.ToTensor()
elif network <= 7:
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
elif network <= 11:
    size = EfficientNet.get_image_size('efficientnet-b{}'.format(network-8))
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor()
    ])
else:
    sys.exit(1)

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
    batch_size=args.batch,
    shuffle=True,
    num_workers=2
)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_data,
    batch_size=args.batch,
    shuffle=False,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=args.batch,
    shuffle=False,
    num_workers=2
)


#Lenet
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
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

#Lenet with regularization
class Lenet_r(nn.Module):
    def __init__(self):
        super(Lenet_r, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1*1*120)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def init_weights(m): # init with 0
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0)
        torch.nn.init.constant_(m.bias, 0)

# choose network
if network == 0:
    net = Lenet()
    # net.apply(init_weights)
elif network == 1:
    net = Lenet_r()
elif network == 2:
    net = models.mobilenet_v2(num_classes=10)
elif network == 3:
    net = models.vgg16(num_classes=10)
elif network == 4:
    net = models.vgg16_bn(num_classes=10)
elif network == 5:
    net = models.vgg16_bn(num_classes=10, init_weights=False)
elif network == 6:
    net = models.resnet50(num_classes=10)
elif network == 7:
    net = models.resnet101(num_classes=10)
elif network <= 11:
    net = EfficientNet.from_name('efficientnet-b{}'.format(network-8), override_params={'num_classes': 10})
else:
    sys.exit(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.para and torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)


# add noise to grad
class SGD_with_noise(optim.SGD):
    @torch.no_grad()
    def step(self, t, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                sigma = 0.01 / ((1+t)**0.55) ##
                noise = torch.empty_like(p.grad).normal_(mean=0, std=sigma) ##
                d_p = p.grad + noise # add noise to grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(d_p, alpha=-group['lr'])
        return loss

@torch.no_grad()
def add_noise(net, a, list):  # add noise to weight
    for p in net.parameters():
        noise = torch.empty_like(p).uniform_(-a, a)
        p.add_(noise)
        list.append(noise)

@torch.no_grad()
def denoise(net, list):  # remove noise from weight
    for (p, noise) in zip(net.parameters(), list):
        p.add_(noise, alpha=-1)

criterion = nn.CrossEntropyLoss()
if args.noise: # grad_noise
    optimizer = SGD_with_noise(net.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.m, weight_decay=args.wd)
# learning rate decay
if args.lrdecay == 0:
    scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=args.dr) #reduce every a steps
elif args.lrdecay == 1:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.dr) # reduce every step
elif args.lrdecay == 2:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.dr, patience=3, verbose=True) # reduce when stop increasing


if args.load:
    net.load_state_dict(torch.load(model_file))

print(args.network)

#training
num_epochs = args.epoch

train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
elapsed_time = 0
counts = 0

for epoch in range(num_epochs):
    start = time.time()
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

    #train
    net.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  #initialize
        if args.smooth:  #smoothout
            noise_list = []
            add_noise(net, 0.03, noise_list)
        outputs = net(inputs)  #output
        loss = criterion(outputs, labels)  #loss
        train_loss += loss.item()
        acc = (outputs.max(1)[1] == labels).sum()
        train_acc += acc.item()
        loss.backward()  #backward
        if args.smooth:  #smoothout
            denoise(net, noise_list)
        if args.noise:  #grad_noise
            optimizer.step(t=counts)  #update weight
        else:
            optimizer.step()
        counts += 1
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
    if args.lrdecay == 2:
        scheduler.step(avg_val_acc)
    elif args.lrdecay >= 0:
        scheduler.step()

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

if not args.test:
    # write log
    logging.basicConfig(filename='net_logger.log', level=logging.INFO)
    #logging.info('Using {} with lr: {}, epochs: {}, m: {}, wd: {}'
    #        .format(args.network, args.lr, args.epoch, args.m, args.wd))
    logging.info(sys.argv)
    logging.info('tl: {}'.format(train_loss_list))
    logging.info('ta: {}'.format(train_acc_list))
    logging.info('vl: {}'.format(val_loss_list))
    logging.info('va: {}'.format(val_acc_list))
    logging.info('elapsed time: {}, accuracy: {}'.format(timeperepoch, accuracy))

    # save model
    torch.save(net.state_dict(), model_file)

    # plot
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
