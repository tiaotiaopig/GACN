from keras import backend as K
from keras.activations import relu
from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,Activation,PReLU,LeakyReLU,MaxoutDense
from keras.optimizers import Adam,RMSprop
from keras import initializers
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
# import deal_data as d
import csv
from keras import backend, losses

randomDim = 350

def T_F_and(t1, t2):
    return t1 and t2

def get_X(arr, keys):
    df = pd.DataFrame(arr, columns = keys)
    # print('transed df:', df)
    X=d.trans(df, keys)
    # print(X.shape)
    return X, X.shape[1]
#print(X_train.shape[0])


def initNormal(shape, name=None):
    return initializers.normal(shape) #删去后面scale=0.2,  name=name

adam = Adam(lr=0.0002, beta_1=0.5)
#
def wasserstein_loss(y_true, y_pred):
    """
    Wasserstein distance for GAN
    author use:
    g_loss = mean(-fake_logit)
    c_loss = mean(fake_logit - true_logit)
    logit just denote result of discrimiantor without activated
    """
    return K.mean(y_true * y_pred)


# Build Generative Model
def build_model(input_shape, rad = randomDim):
    global randomDim
    randomDim = rad
    g_input = Input(shape=(randomDim,))
    # '''here the initNormal can equal to the 'normal',init can receive funciton'''

    H = Dense(512)(g_input)
    H = LeakyReLU(0.2)(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    # H = Dense(256)(H)
    # H = LeakyReLU(0.2)(H)
    # H = Dense(256)(H)
    # H = LeakyReLU(0.2)(H)

    # Because train data have normalized to [-1,1] ,tanh can be fit
    g_output = Dense(input_shape, activation='linear')(H)  #784 改17
    generator = Model(g_input, g_output)
    # generator.compile(loss=wasserstein_loss,optimizer='RMSprop')


    # Build Discriminative Model
    d_input = Input(shape=(input_shape,))#784 改17
    D = Dense(256)(d_input)# 删去, init=initNormal(17)
    D = LeakyReLU(0.2)(D)
    # D = Dropout(0.3)(D)
    D = Dense(56)(D)
    # D = LeakyReLU(0.2)(D)
    # D = Dropout(0.3)(D)
    # D = Dense(256)(D)
    # D = LeakyReLU(0.2)(D)
    # D = Dropout(0.3)(D)
    # D = Dense(256)(D)
    # D = LeakyReLU(0.2)(D)

    d_output = Dense(1, activation='linear')(D)

    discriminator = Model(d_input, d_output)

    discriminator.compile(loss=losses.binary_crossentropy, optimizer='sgd')

    # Build Dcoo Model
    d_input = Input(shape=(input_shape,))#784 改17
    Dc = Dense(256)(d_input)# 删去, init=initNormal(17)
    Dc = LeakyReLU(0.2)(Dc)
    Dc = Dense(256)(Dc)
    # Dc = LeakyReLU(0.2)(Dc)
    # Dc = Dropout(0.3)(Dc)
    # Dc = Dense(256)(Dc)
    # Dc = LeakyReLU(0.2)(Dc)
    # Dc = Dropout(0.3)(Dc)
    # Dc = Dense(256)(Dc)
    # Dc = LeakyReLU(0.2)(Dc)

    dc_output = Dense(1, activation='linear')(Dc)

    discriminator_coo = Model(d_input, dc_output)

    discriminator_coo.compile(loss=losses.binary_crossentropy, optimizer='sgd')
    # print(generator.summary())
    # print(discriminator.summary())

    # Combine the two networks
    discriminator.trainable = False
    gan_input = Input((randomDim,))
    x = generator(gan_input)
    # print(x)
    gan_output = discriminator(x)

    gan = Model(gan_input, gan_output)
    # gan.compile(loss=losses.categorical_crossentropy, optimizer='sgd')

    gan_input = Input((randomDim,))
    x = generator(gan_input)
    # print(x)
    gan_output_coo = discriminator_coo(x)

    gan_coo = Model(gan_input, gan_output_coo)
    gan_coo.compile(loss=losses.binary_crossentropy, optimizer='sgd')



    return gan, gan_coo, generator, discriminator, discriminator_coo



# Plot the loss from each epoch
def plot_loss(epoch, Dloss, Gloss):
    plt.figure(figsize=(10, 8))
    plt.plot(Dloss, label='Dsicriminiative loss')
    plt.plot(Gloss, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('wgan_loss_epoch_%d.png' % epoch)


# Create a wall of generated MNIST images
def plotGeneratedImages(generator, fileheader, example=4000, dim=(10, 10), figsize=(10, 10)):#输出

    noise = np.random.normal(0, 1, size=(example, randomDim))
    fake = generator.predict(noise)
    print('fake shape:', fake.shape)
    fakeout = np.array(d.detrans(fake, fileheader))
    print('fakeout shape:', fakeout.shape)
    return fakeout


def saveModels(epoch):
    if not os.path.exists('Model_para'):
        os.mkdir('Model_para')
    generator.save('Model_para/models_wgan_generated_epoch_%d.h5' % epoch)
    discriminator.save('Model_para/models_wgan_discriminated_epoch_%d.h5' % epoch)


def clip_weight(weight, lower, upper):
    weight_clip = []
    for w in weight:
        w = np.clip(w, lower, upper)
        weight_clip.append(w)
    return weight_clip

def kmeans_for_train():
    pass

def loss_for_GAN(y_true, y_pred_GAN):
    global loss_gancoo
    # print(loss_gancoo)
    return losses.binary_crossentropy(y_true, y_pred_GAN) + 0.5 * loss_gancoo

loss_gancoo = 0
def train(gan, gan_coo, generator, discriminator, discriminator_coo, X_train, X_side = None, epochs=1, batchsize=500):
    global loss_gancoo
    batchCount = X_train.shape[0] / batchsize
    print('Epochs', epochs)
    print('Bathc_size', batchsize)
    print('Batches per epoch', batchCount)
    Dloss = []
    Gloss = []
    # range ande xrange the different is a list and a generator
    for e in range(1, epochs + 1):
        dloss=0
        gloss=0
        # print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(round(batchCount))):  #增加取整
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchsize, randomDim])

            #已有数据的随机采样
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchsize)]

            # generate fake MNIST images
            generatedImages = generator.predict(noise)

            X = np.concatenate([imageBatch, generatedImages])
            # Labels for generated and real data
            yDis = np.ones(2 * batchsize)
            # one-sided label smoothing
            yDis[:batchsize] = -1

            # Train discriminator
            # gan.compile(loss=wasserstein_loss, optimizer='RMSprop')
            discriminator.trainable = True

            dloss = discriminator.train_on_batch(X, yDis)

            # Train discriminator_coo
            # discriminator_coo.trainable = True
            if X_side is not None:
                sideBatch = X_side[np.random.randint(0, X_side.shape[0], size=batchsize)]
                X_side_fake = np.concatenate([sideBatch, generatedImages])
                y_true_coo = np.ones(2 * batchsize)
                y_true_coo[:batchsize] = -1
                discriminator_coo.train_on_batch(X_side_fake, y_true_coo)
                # #在side方向上的loss
                # loss = losses.categorical_crossentropy(y_side, gan_coo.output)
                # #在side方向上的梯度
                # gradient_for_side = backend.gradients(loss, gan_coo.input)
                # print(gradient_for_side)

                # Train generator
                loss_gancoo = losses.binary_crossentropy(-1, gan_coo.predict(noise))

                noise = np.random.normal(0, 1, size=[batchsize, randomDim])
                gan.compile(loss = loss_for_GAN, optimizer = 'sgd')
                # gan.compile(loss=losses.binary_crossentropy, optimizer='sgd')
                yGen = np.ones(batchsize) * -1
                discriminator.trainable = False

                gloss = gan.train_on_batch(noise, yGen)
            else:
                noise = np.random.normal(0, 1, size=[batchsize, randomDim])
                # gan.compile(loss = loss_for_GAN, optimizer = 'sgd')
                gan.compile(loss=losses.binary_crossentropy, optimizer='sgd')
                yGen = np.ones(batchsize) * -1
                discriminator.trainable = False
                gloss = gan.train_on_batch(noise, yGen)



            '''
            d_weight = discriminator.get_weights()
            d_weight = clip_weight(d_weight,-0.01,0.01)
            discriminator.set_weights(d_weight)
            '''
        # Store loss of most recent batch from this epoch

        Dloss.append(dloss)
        # print(dloss)
        Gloss.append(gloss)

        # if e == 1 or e % 100 == 0:
        #     plotGeneratedImages(keys, e)
        #     saveModels(e)
    # plot_loss(e, Dloss, Gloss)


if __name__ == '__main__':

    X_train = get_X(df)

    train(keys, X_train, 400,500)



