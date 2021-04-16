from keras2 import *
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tools import sort_by_keys
from keras import losses
from keras.utils import to_categorical
from keras.optimizers import Adam
from deal_fashion_mnist import *
def round_ (s):
    return round(s, 2)
def sample_from_MNOST(l):
    label_list = [2,4]
    if l in label_list:
        return True
    else:
        return False

def MNOST_01(l):
    if l == 2:
        return 0
    else:
        return 1

def visul_1_dim(X_train, show = False):
    l = X_train.reshape(1, -1)[0]
    # print(l)
    dict_sta = dict(Counter(list(map(round_,l))))
    # X_train = X_train.reshape((-1, 1))
    # print(X_train)
    x_show, yshow = sort_by_keys(dict_sta)
    plt.plot(x_show, yshow)
    # if show:
    #     plt.show()

def visul_1_dim_scatter(X_train, y):
    xnum = X_train[:, 0]
    print(list(xnum))
    print(len(list(xnum)))
    ynum = np.ones(xnum.shape[0]) * y
    plt.scatter(xnum, ynum, s = 1, c = 'orange')

# X_train = np.random.uniform(1,2, (1000, 2))
# X_side1 = np.hstack((np.random.uniform(0,3, 3000).reshape((-1,1)), np.random.uniform(0,1, 3000).reshape((-1,1))))
# # X_side2 = np.hstack((np.random.uniform(0,3, 3000).reshape((-1,1)), np.random.uniform(2,3, 3000).reshape((-1,1))))
# X_side3 = np.hstack((np.random.uniform(0,1, 1000).reshape((-1,1)), np.random.uniform(1,2, 1000).reshape((-1,1))))
# # X_side4 = np.hstack((np.random.uniform(2,3, 1000).reshape((-1,1)), np.random.uniform(1,2, 1000).reshape((-1,1))))
# X_side = np.vstack((X_side1, X_side3))
# # print(X_train)
# # plt.scatter(X_train[:, 0], X_train[:, 1], s = 1)
# # plt.scatter(X_side[:, 0], X_side[:, 1], s = 1)
def train_catog_model(X_total, y_total):
    model = Sequential()
    model.add(Dense(512, activation="sigmoid", name='dense_1'))
    model.add(Dense(512, activation="sigmoid", name='dense_2'))
    # model.add(Dense(1024, activation="sigmoid", name='dense_3'))
    # model.add(Dense(1024, activation="sigmoid", name='dense_4'))
    # model.add(Dense(128, activation="sigmoid", name='dense_5'))
    model.add(Dense(1, activation="sigmoid", name='dense_6'))
    # Compile model
    optimizer= Adam(0.00002, 0.5)
    model.compile(optimizer = optimizer, loss=losses.binary_crossentropy, metrics=["accuracy"])
    history = model.fit(X_total, y_total, validation_split=0.33, epochs=120, batch_size=80, verbose=1)
    model.save(r'bi_01_model_for_fashion.h5')
    return model


##生成数据训练
sam_num = 3000
X_train = np.random.uniform(100,110,sam_num)
X_train = X_train.reshape((-1, 1))
y_train = np.zeros(X_train.shape[0]).reshape((-1, 1))

X_side_1 = np.random.uniform(50,100,int(sam_num )).reshape((-1, 1))
y_side_1 = (np.ones(X_side_1.shape[0]) * 1).reshape((-1, 1))

# X_side_2 = np.random.normal(100,200,sam_num).reshape((-1, 1))
# y_side_2 = (np.ones(X_side_2.shape[0]) * 3).reshape((-1, 1))

# X_side_3 = np.random.uniform(100,150,int(sam_num / 2)).reshape((-1, 1))
# y_side_3 = (np.ones(X_side_3.shape[0]) * 1).reshape((-1, 1))


train_images = load_train_images('fashion')
train_labels = load_train_labels('fashion')

# X_total = np.vstack((X_train, X_side_1))
# y_total = np.vstack((y_train, y_side_1))
# y_total = to_categorical(y_total)[:, 1:]
sample_num = 10000
shuffle_idx = np.arange(sample_num)
np.random.shuffle(shuffle_idx)

print(train_images.shape)
print(train_labels.shape)
print(train_labels)

sample_list = list(map(sample_from_MNOST, train_labels))

train_images = train_images[sample_list]
train_labels = train_labels[sample_list]

print(train_images.shape)
print(train_labels.shape)
print(train_labels)


X_total = train_images[shuffle_idx]
X_total = X_total.reshape((X_total.shape[0], X_total.shape[1]* X_total.shape[2]))
y_total = train_labels[shuffle_idx]

#处理ytotal 转换为2分类
y_total = np.array(list(map(MNOST_01, y_total))).reshape((-1, 1))
print(Counter(y_total[:,0]))


print(X_total.shape)
model_bi = train_catog_model(X_total, y_total)

##GACN

# X_side = np.vstack((X_side_1, X_side_2, X_side_3))
# y_side = np.vstack((y_side_1, y_side_2, y_side_3))

get_train_from_mix = y_total[:,0] == 0
get_side_from_mix = y_total[:,0] == 1

gan = GAN(X_train = X_total[get_train_from_mix], X_side =None)
gan.train(epochs=800, batch_size=1000, sample_interval=5)
noise = np.random.normal(0, 1, (2000, 100))  #输入噪声样本 dim
# fake = gan.generator.predict(noise)
c = 0
list_GAN_process = gan.GAN_process
list_GAN_process = np.array(list_GAN_process)
list_GAN_process = list_GAN_process.reshape((list_GAN_process.shape[0], -1))
list_GAN_process.astype(np.float32)
# print(list_GAN_process)

# gan.discriminator_coo.save(r'D:\Experi\GACN\MNIST\GACN_from_coo.h5')
#清空内存
X_total = []
get_side_from_mix = []
train_images = []
train_labels = []

np.save(r'X_side_bina_process_GAN_fashion', list_GAN_process)



# y_max = len(list_GAN_process)
# print('y_max', y_max)
# for fakes in list_GAN_process:
#     fakes = np.array(fakes)
#     visul_1_dim_scatter(fakes, y)
#     c += 1
# plt.ylim(0, 1.3)
# plt.show()



#
# visul_1_dim(X_train, show = True)
# visul_1_dim(fake, show = True)
# plt.show()


