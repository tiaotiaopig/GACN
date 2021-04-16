from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import random
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from keras import optimizers
from keras import losses
from simple_keras import *
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import time
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        # _val_recall = recall_score(val_targ, val_predict)
        # _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall)
        # self.val_precisions.append(_val_precision)
        # print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        # print(' — val_f1:' ,_val_f1)
        # print('— val_f1: %f — val_precision: %f ' %(_val_f1, _val_precision, ))
        # return



def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 30 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


def GACN(X, y, coo = True):
    X_aug = np.zeros(X.shape[1])
    y_aug = [0]

    labels_for_col = y
    labels_set = set(labels_for_col)
    # X = np.array(X)
    for label in labels_set:
        #获取该类别
        X_intra_small = X[labels_for_col == label, :]
        #先将真实数据放入Aug中
        X_aug = np.vstack((X_aug, X_intra_small))
        #对应的y
        y_aug = y_aug + X_intra_small.shape[0] * [label, ]
        # Gan = True
        if label == 'BENIGN':
            continue
        print(label)
        #side
        X_side = X[labels_for_col != label, :]
        gan, gan_coo, generator, discriminator, discriminator_coo = build_model(X_intra_small.shape[1], rad = 100)
        if coo:
            train(gan, gan_coo, generator, discriminator, discriminator_coo, X_intra_small, X_side=X_side, epochs=100,
                  batchsize=200)
        else:#compare
            train(gan, gan_coo, generator, discriminator, discriminator_coo, X_intra_small, X_side=None, epochs=100,
                  batchsize=200)
        noise = np.random.normal(0, 1, size=(3000, 100))
        #生成X放入Aug
        X_fake = generator.predict(noise)
        X_aug = np.vstack((X_aug, X_fake))
        #y
        y_aug = y_aug + X_fake.shape[0] * [label, ]
    y_aug = np.array(y_aug)
    return X_aug[1:,], y_aug[1:]
    # return np.hstack(X_aug[1:,], y_aug[1:])




def col_hstack_list(a,b):
    return np.hstack((a, b))

def sample_for_benign(lab):
    return_list = []
    if lab == 'BENIGN' and random.random() <=  1/80:
        return True
    if lab == 'DoS Hulk' and random.random() <=  1/40:
        return True
    if lab == 'DoS GoldenEye' and random.random() <= 1 / 2:
        return True
    if lab == 'DoS slowloris' :
        return True
    if lab == 'DoS Slowhttptest' :
        return True
    else:
        return False

def scale_for_X(r):
    enc = OneHotEncoder(sparse=False, categories='auto')
    df = pd.read_csv(r"D:\Experi\feature_vec_total.csv")
    print("df1.shape:",df.shape)
    X1 = np.array(df.iloc[1:, :df.shape[1]-1])
    y1 = np.array(df.iloc[1:, -1]).reshape(-1,1)
    y1_comp = np.array(df.iloc[1:, -1])
    #删除bengin
    sample_list_01 = np.array(list(map(sample_for_benign, y1)))
    X1 = X1[sample_list_01, :]
    y1 = y1[sample_list_01, :]
    # print(y1.reshape(1,-1))
    print("Counter y1",Counter(list(y1.reshape(1,-1)[0])))
    # y1 = enc.fit_transform(y1)

    small_sample_list = np.random.randint(0, X1.shape[0], r)
    X1 = X1[small_sample_list, :]
    y1 = y1[small_sample_list, :]

    # feature_vec_total = np.array(list(map(col_hstack_list, X1, y1)))
    feature_vec_total_small = np.hstack((X1, y1))
    np.savetxt(r'D:\Experi\feature_vec_total_small.csv', feature_vec_total_small, delimiter=",", fmt='%s')
    print("saved successfully!")

def train_and_plot(X,y, change_lr = False):
    model = Sequential()
    # model.add(Dense(12, activation="sigmoid", name='dense_1'))
    model.add(Dense(256, activation="sigmoid", name='dense_2'))
    model.add(Dense(256, activation="sigmoid", name='dense_3'))
    model.add(Dense(256, activation="sigmoid", name='dense_add'))
    model.add(Dense(y.shape[1], activation="sigmoid", name='dense_4'))
    # Compile model
    if change_lr:
        print("ASDASD")
        sgdx2 = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgdx2, loss='categorical_crossentropy_X2', metrics=["accuracy"])
        # model.compile(optimizer = sgdx2, loss=categorical_crossentropy_X2, metrics=["accuracy"])
    else:
        x_train,y_train,x_label,y_label = train_test_split(X, y, train_size=0.8, random_state=233)
        # reduce_lr = LearningRateScheduler(scheduler)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

        # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        adm = optimizers.Adam()
        metrics = Metrics()
        model.compile(optimizer = adm, loss="categorical_crossentropy", metrics=["accuracy"])

    # Fit the model
    history = model.fit(X, y, validation_split=0.33, nb_epoch=100, batch_size=20, verbose=1, callbacks=[metrics,]) # list all data in history
    print('time spend:', time.clock() - ts)
    print(history.history.keys())
    model.reset_states()
    print("F1:", np.mean(metrics.val_f1s))
    return history.history['accuracy'], history.history['loss'], \
           history.history['val_accuracy'],history.history['val_loss']
    # return  history.history['val_accuracy'],history.history['val_loss']

# enc = OneHotEncoder(sparse=False, categories='auto')
# df = pd.read_csv(r"D:\Experi\feature_vec_total_small.csv")
#
# X1 = np.array(df.iloc[1:, :df.shape[1]-1])
# y1 = np.array(df.iloc[1:, -1]).reshape(-1,1)
# small_sample_list_twice = np.random.randint(0, X1.shape[0], 800)
# X1 = X1[small_sample_list_twice, :]
# y1 = y1[small_sample_list_twice, :]
# print("df1.shape:",df.shape)

# y1 = enc.fit_transform(y1)
# # acc1, loss1 ,val_acc1, val_loss1 = train_and_plot(X1, y1)
# # plt.plot(val_acc1)
# # plt.plot(val_loss1)
# # plt.show()
def sample_B(l):
    if l == 'BENIGN':
        return True
    elif random.random() <= 0.05:
        return True
    else:
        return False


# scale_for_X(20000)


df = pd.read_csv(r"D:\Experi\feature_vec_total_small.csv")
df = df.iloc[1:, :]
label_list_ori  = df.iloc[:, -1]
df = df[list(map(sample_B, label_list_ori))]

# df = df.iloc[small_sample_list_twice,:]
X_ori = np.array(df.iloc[1:, :df.shape[1]-1])
y_ori = np.array(df.iloc[1:, -1])
print(Counter(y_ori))
# enc = OneHotEncoder(sparse=False, categories='auto')
# y1 = y_ori.reshape(-1, 1)
# y1 = enc.fit_transform(y1)
# acc1, loss1 ,val_acc1, val_loss1 = train_and_plot(X_ori, y1)
#
# X_ang, y_aug = (GACN(X_ori,y_ori))
# print("Counter yang_coo",Counter(list(y_aug)))
# enc = OneHotEncoder(sparse=False, categories='auto')
# y_aug = y_aug.reshape(-1, 1)
# y_aug = enc.fit_transform(y_aug)
# #shuffle
# index = np.arange(X_ang.shape[0])
# np.random.shuffle(index)
# X_ang = X_ang[index]
# y_aug = y_aug[index]
#
# acc2, loss2 ,val_acc2, val_loss2 = train_and_plot(X_ang, y_aug)
ts = time.clock()
##########
X_ang, y_aug = GACN(X_ori,y_ori, coo=True)
##################
# X_ang = X_ori
# y_aug = y_ori
print("Counter yang_without_coo",Counter(list(y_aug)))
enc = OneHotEncoder(sparse=False, categories='auto')
y_aug = y_aug.reshape(-1, 1)
y_aug = enc.fit_transform(y_aug)
#shuffle
index = np.arange(X_ang.shape[0])
np.random.shuffle(index)
X_ang = X_ang[index]
y_aug = y_aug[index]

acc3, loss3 ,val_acc3, val_loss3 = train_and_plot(X_ang, y_aug)

plt.plot(val_acc1, label = 'acc1')
plt.plot(val_acc2, label = 'acc2')
plt.plot(val_acc3, label = 'acc3')
plt.legend()
plt.show()
plt.cla()

plt.plot(val_loss1, label = 'val_loss1')
plt.plot(val_loss2, label = 'val_loss2')
plt.plot(val_loss3, label = 'val_loss3')
plt.legend()
plt.show()
plt.cla()