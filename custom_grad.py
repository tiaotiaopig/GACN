
from keras.models import Sequential
from keras.layers import Dense, Dropout
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
from keras.callbacks import Callback
from sklearn.model_selection import cross_val_score
from keras.optimizers import SGD_custom, SGD_custom_pre

import tensorflow as tf









X_1 = np.random.uniform(0,0.5,20000)
X_1 = X_1.reshape((-1, 10))

X_2 = np.random.uniform(0.5,1,20000)
X_2 = X_2.reshape((-1, 10))

y_labels = np.ones(4000) - 1
y_labels[2000:] = np.ones(2000)
y_labels = y_labels.reshape(-1, 1)
X_train = np.vstack((X_1, X_2))
print(X_train.shape)
print(y_labels)
sgd_1 = SGD_custom_pre()

def get_coo_gradients_():
    global v_list
    return v_list

# class OptGain(Callback):
#     def on_epoch_end(self, epoch = 2, logs = {}):
#         opt = self.model.optimizer
#         val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
#         val_targ = np.argmax(self.validation_data[1], axis=1)
#         loss = losses.binary_crossentropy(val_predict, val_targ)
#         params = self.model.weights
#         print(opt.get_gradients(loss, params))
#







model = Sequential()
model.add(Dense(12, activation="sigmoid", name='dense_1'))
# model.add(Dropout(0.9))
# model.add(Dense(12, activation="sigmoid", name='dense_2'))
model.add(Dense(1, activation="sigmoid", name='dense_4'))
# Compile model
model.compile(optimizer = sgd_1, loss='binary_crossentropy', metrics=["accuracy"])
# Fit the model
# metrics = Metrics()
# opt = OptGain()

for i in range(200):
    idx = np.random.randint(0, X_train.shape[0], 10)
    X_train_epoch = X_train[idx]
    y_labels_epoch = y_labels[idx]
    model.train_on_batch(X_train_epoch, y_labels_epoch)
    weight = model.weights



# model.train_on_batch()
history = model.fit(X_train, y_labels, validation_split=0.33, epochs=30, batch_size=128, verbose=0) # list all data in history

w = (model.get_weights())

unk = np.random.uniform(0,0.5,10).reshape(1, -1)
print(unk)
print(unk.shape)
print(model.predict(unk))
v_list = sgd_1.get_massage_to_my_file()
print(v_list)
#
# with tf.compat.v1.Session() as sess:
#     print(sess.run(v_list[0]))

# sgd_2 = SGD_custom()
# sgd_2.get_coo_gradients = get_coo_gradients_
# get_coo_gradients_()
# print(sgd_2.get_coo_gradients)
model.reset_states()
print(w)
model.set_weights(w)
print(model.summary())

#
#
# model.compile(optimizer = sgd_2, loss='binary_crossentropy', metrics=["accuracy"])
# history = model.fit(X_train, y_labels, validation_split=0.33, epochs=30, batch_size=128, verbose=0,)

