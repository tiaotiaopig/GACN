from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD, SGD_custom
import keras
import numpy as np
from keras import backend, losses
from keras.layers import Input,Dense,Dropout,BatchNormalization,Activation,PReLU,LeakyReLU,MaxoutDense
from keras.models import Model

d_input = Input(shape=(13,))
D = Dense(25)(d_input)
D = Dropout(0.3)(D)
d_output = Dense(10, activation='linear')(D)
model = Model(d_input, d_output)


# 生成虚拟数据
x_train = np.random.random((200, 13))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(200, 1)), num_classes=10)
x_test = np.random.random((200, 13))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(200, 1)), num_classes=10)

# 使用顺序模型搭建网络
# model = Sequential()
#
# model.add(Dense(13, activation='relu'))  # 全连接
# model.add(Dropout(0.5))  # 随机失活层
# model.add(Dense(10, activation='softmax'))

# 优化
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)


# 整合模型
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 训练
model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=0)

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()


score = model.evaluate(x_test, y_test, batch_size=128)
loss = losses.categorical_crossentropy(y_test, model.output)
for name, weight in zip(names, weights):
    print(name, weight.shape)
    gradient = backend.gradients(loss, model.inputs)
    print(gradient)
print(score)

def our_get_gradients(a):
    return gradient

sgd = SGD_custom(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
SGD_custom.get_coo_gradients = our_get_gradients

d_input = Input(shape=(13,))
D = Dense(25)(d_input)
D = Dropout(0.3)(D)
d_output = Dense(10, activation='linear')(D)
model = Model(d_input, d_output)

# 整合模型
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# 训练
inputs = backend.placeholder(shape=(1, 13))
model.inputs = inputs
model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=0   )
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)