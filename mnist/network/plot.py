import matplotlib.pyplot as plt

accuracy = [98.38, 99.15, 99.23, 99.1, 98.88]
time = [2.64, 139.61, 218.06, 95.33, 256.7]
net = ["Lenet", "Mobilenet", "VGG", "Resnet", "Efficientnet"]

fig = plt.figure()

for (i,j,k) in zip(time,accuracy,net):
        plt.plot(i,j,'o')
        plt.annotate(k, xy=(i, j))
plt.xlabel('time')
plt.ylabel('accuracy')
plt.title('Time and Accuracy')
plt.grid()
fig.savefig("net_mnist.png")
