import matplotlib.pyplot as plt

ta_Lenet = [0.105475, 0.11065, 0.110525, 0.109825, 0.14695, 0.218275, 0.259075, 0.270725, 0.2842, 0.299775, 0.318225, 0.33875, 0.3565, 0.37595, 0.39065, 0.4067, 0.4157, 0.4313, 0.440775, 0.448775, 0.459875, 0.468275, 0.474725, 0.48155, 0.490075, 0.49295, 0.498375, 0.50495, 0.509375, 0.514675, 0.51805, 0.522825, 0.527825, 0.532675, 0.537225, 0.53885, 0.54685, 0.547275, 0.553675, 0.5577, 0.563025, 0.567625, 0.5703, 0.57465, 0.580175, 0.582475, 0.58605, 0.58925, 0.594625, 0.5952, 0.6016, 0.6045, 0.60775, 0.6123, 0.61405, 0.617325, 0.6195, 0.623275, 0.62935, 0.627875, 0.632325, 0.633925, 0.6366, 0.640475, 0.642125, 0.64465, 0.646225, 0.650875, 0.655875, 0.656425, 0.658325, 0.6614, 0.6609, 0.6635, 0.6682, 0.6712, 0.675625, 0.67825, 0.677425, 0.678625, 0.682725, 0.6834, 0.685525, 0.68975, 0.689125, 0.6923, 0.69435, 0.6958, 0.699, 0.698475, 0.703925, 0.7031, 0.708675, 0.7096, 0.710525, 0.713825, 0.713275, 0.717825, 0.717825, 0.722225]
va_Lenet = [0.1161, 0.1036, 0.1156, 0.1125, 0.1859, 0.2594, 0.2568, 0.2621, 0.2945, 0.3072, 0.3258, 0.3518, 0.3683, 0.3906, 0.3996, 0.3815, 0.4145, 0.4147, 0.4392, 0.4481, 0.4569, 0.454, 0.4584, 0.4661, 0.4644, 0.479, 0.4895, 0.4852, 0.497, 0.4952, 0.5057, 0.505, 0.4934, 0.513, 0.5234, 0.5246, 0.5314, 0.5257, 0.5246, 0.5335, 0.5389, 0.5368, 0.5436, 0.5485, 0.5427, 0.5524, 0.5342, 0.5531, 0.5578, 0.5611, 0.5477, 0.5548, 0.5679, 0.5653, 0.566, 0.5684, 0.5836, 0.5758, 0.5758, 0.5645, 0.5795, 0.5734, 0.5851, 0.5743, 0.5871, 0.582, 0.581, 0.5972, 0.5896, 0.5921, 0.5975, 0.5897, 0.5944, 0.5979, 0.5924, 0.5983, 0.5871, 0.5964, 0.6042, 0.6053, 0.6003, 0.5987, 0.5829, 0.594, 0.6005, 0.6002, 0.5887, 0.5986, 0.6012, 0.6032, 0.6047, 0.6033, 0.5955, 0.6102, 0.6077, 0.5957, 0.5996, 0.6059, 0.6103, 0.6034]
ta_Mobilenet = [0.2349, 0.37175, 0.453725, 0.51925, 0.569475, 0.603475, 0.629275, 0.66165, 0.687225, 0.7057, 0.721875, 0.740425, 0.759075, 0.77725, 0.78835, 0.802575, 0.82025, 0.83255, 0.843825, 0.859775, 0.8719, 0.88075, 0.8882, 0.900175, 0.9084, 0.916675, 0.924425, 0.928875, 0.936825, 0.943825, 0.950875, 0.958425, 0.96115, 0.96605, 0.965375, 0.968675, 0.9699, 0.973525, 0.97535, 0.978425, 0.980075, 0.98105, 0.983725, 0.9845, 0.984925, 0.9864, 0.988, 0.98845, 0.989275, 0.9902]
va_Mobilenet = [0.2752, 0.4224, 0.4941, 0.547, 0.5632, 0.5879, 0.625, 0.6625, 0.6731, 0.6831, 0.6995, 0.7189, 0.7064, 0.7244, 0.7263, 0.7345, 0.7449, 0.7428, 0.747, 0.7473, 0.7571, 0.7472, 0.756, 0.7591, 0.7559, 0.7503, 0.7543, 0.7623, 0.7558, 0.7464, 0.7538, 0.753, 0.7522, 0.7582, 0.7627, 0.759, 0.7603, 0.7657, 0.76, 0.7634, 0.7647, 0.766, 0.7634, 0.7666, 0.7666, 0.7689, 0.7663, 0.7732, 0.7611, 0.7698]
ta_VGG16 = [0.338425, 0.46985, 0.52355, 0.5699, 0.61385, 0.644525, 0.6745, 0.6988, 0.72625, 0.747475, 0.767525, 0.789225, 0.80835, 0.82785, 0.843775, 0.8605, 0.876025, 0.8913, 0.903975, 0.920025, 0.928625, 0.937775, 0.94845, 0.955675, 0.96145, 0.96785, 0.9721, 0.974275, 0.98025, 0.98255, 0.982775, 0.985375, 0.98615, 0.9859, 0.989175, 0.98955, 0.990575, 0.991175, 0.993475, 0.992825, 0.9931, 0.99375, 0.99435, 0.9938, 0.995, 0.994375, 0.995275, 0.9956, 0.995125, 0.996475]
va_VGG16 = [0.4637, 0.5219, 0.5663, 0.5926, 0.6278, 0.6453, 0.6648, 0.6866, 0.6964, 0.7242, 0.7281, 0.7371, 0.7394, 0.7501, 0.7595, 0.744, 0.762, 0.7595, 0.7657, 0.78, 0.7817, 0.7799, 0.762, 0.7641, 0.7767, 0.7758, 0.7748, 0.7843, 0.7633, 0.7834, 0.7761, 0.7825, 0.7519, 0.7829, 0.7885, 0.7891, 0.7879, 0.7914, 0.7741, 0.7869, 0.7962, 0.798, 0.7962, 0.788, 0.7938, 0.7966, 0.7889, 0.7821, 0.7984, 0.7961]
ta_Resnet50 = [0.20065, 0.289975, 0.345425, 0.383275, 0.420875, 0.46035, 0.483875, 0.505525, 0.5304, 0.5484, 0.571075, 0.588275, 0.6073, 0.6277, 0.64225, 0.6619, 0.67475, 0.692175, 0.7112, 0.7275, 0.7409, 0.759725, 0.777375, 0.79795, 0.814575, 0.8305, 0.846925, 0.862025, 0.8725, 0.894, 0.904225, 0.91575, 0.9281, 0.943275, 0.946025, 0.955975, 0.9583, 0.9628, 0.9665, 0.968725, 0.9758, 0.97475, 0.977375, 0.980475, 0.983525, 0.984375, 0.9839, 0.984775, 0.986325, 0.9859]
va_Resnet50 = [0.2338, 0.3214, 0.3614, 0.3982, 0.4417, 0.4367, 0.4916, 0.5044, 0.5169, 0.5125, 0.5589, 0.5574, 0.5963, 0.584, 0.5764, 0.6038, 0.5993, 0.6169, 0.6237, 0.5876, 0.6179, 0.5173, 0.6347, 0.6422, 0.6452, 0.6391, 0.6121, 0.6423, 0.6555, 0.6464, 0.6604, 0.6715, 0.6699, 0.6726, 0.658, 0.6659, 0.6601, 0.6655, 0.6834, 0.6763, 0.6743, 0.6764, 0.6818, 0.6727, 0.6827, 0.6951, 0.6837, 0.6842, 0.674, 0.6896]
ta_Efficientnetb1 = [0.251775, 0.404225, 0.4995, 0.569325, 0.62085, 0.6554, 0.695175, 0.725775, 0.75235, 0.779275, 0.801825, 0.82155, 0.83635, 0.855675, 0.870225, 0.881175, 0.893775, 0.906725, 0.91185, 0.921575, 0.926325, 0.93255, 0.939, 0.940025, 0.948725, 0.949825, 0.956275, 0.95575, 0.960425, 0.962, 0.963675, 0.96495, 0.9663, 0.970975, 0.9696, 0.97145, 0.972275, 0.970575, 0.97475, 0.976425, 0.97535, 0.9773, 0.9769, 0.98055, 0.980575, 0.980675, 0.9811, 0.98135, 0.9803, 0.984]
va_Efficientnetb1 = [0.2281, 0.4644, 0.5316, 0.6009, 0.6051, 0.6798, 0.7078, 0.7289, 0.7262, 0.7522, 0.7652, 0.7445, 0.77, 0.7916, 0.7882, 0.7663, 0.7892, 0.7903, 0.784, 0.7915, 0.7947, 0.795, 0.796, 0.798, 0.7927, 0.7983, 0.7891, 0.8067, 0.8039, 0.8062, 0.8135, 0.8052, 0.8006, 0.8103, 0.815, 0.8123, 0.8099, 0.8028, 0.812, 0.8139, 0.8129, 0.8108, 0.8036, 0.8082, 0.8174, 0.8142, 0.8162, 0.8152, 0.811, 0.8084]
num_epochs = 50

ta_list = [ta_Lenet[:50], ta_Mobilenet, ta_VGG16, ta_Resnet50, ta_Efficientnetb1]
va_list = [va_Lenet[:50], va_Mobilenet, va_VGG16, va_Resnet50, va_Efficientnetb1]

colorlist = ['red', 'blue', 'orange', 'green', 'gold']
netlist = ['Lenet', 'Mobilenet', 'VGG16', 'Resnet50', 'EfficientNetb1']


fig1 = plt.figure()
for (train_acc_list, val_acc_list, color, net) in zip(ta_list, va_list, colorlist, netlist):
    plt.plot(range(num_epochs), train_acc_list, color=color, linestyle='-', label=net)
    plt.plot(range(num_epochs), val_acc_list, color=color, linestyle='--')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()
fig1.savefig('net_cifar10_acc.png')