from __future__ import print_function, division

# from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, Reshape, Flatten
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses, backend
from random import random
import matplotlib.pyplot as plt

import sys

import numpy as np
import keras.backend as K
from keras.callbacks import LearningRateScheduler


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)


class GAN():
    def __init__(self, X_train, X_side = None):
        global loss_gancoo
        # self.GAN_process = np.zeros(X_train.shape[1]).reshape(1, -1)
        self.GAN_process = []
        self.loss1_list = []
        self.loss2_list = []
        self.score_mean_list = []
        self.score_pre = 999
        self.judge_count = 0
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100   #输入噪声dim 第二参数
        self.X_train = X_train
        if X_side is None:
            self.is_coo = False
            self.X_side = X_side
        else:
            self.is_coo = True
            self.X_side = X_side

        optimizer_Dadv = Adam(0.0002, 0.5)
        optimizer_GAN = Adam(0.0002, 0.5)
        optimizer_GACN = Adam(0.0002, 0.5)
        optimizer_Dcoo = Adam(0.0002, 0.5)


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_Dadv,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)



        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)


        # if X_side is not None:
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_GAN)

        # else:
        #     self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_GAN)

        #构造Dcoo
        self.discriminator_coo = self.build_discriminator_coo()
        self.discriminator_coo.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_Dcoo,
                                   metrics=['accuracy'])

        validity_coo = self.discriminator_coo(img)
        self.combined_coo = Model(z, validity_coo)
        self.combined_coo.compile(loss='binary_crossentropy', optimizer=optimizer_GACN)

    def return_random_noise(self, sample_num):
        return np.random.normal(0, 1, (sample_num, self.latent_dim))


    def build_generator(self):

        model = Sequential()
        model.add(Dense(28 * 28, input_dim=self.latent_dim))
        model.add(Reshape((28, 28, 1)))
        model.add(Conv2D(32, ((7, 7)), activation = 'sigmoid'))
        model.add(Conv2D(32, ((7, 7)), activation='sigmoid'))
        model.add(Flatten())

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.X_train.shape[1]))


        # model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # model.add(Flatten(input_shape=self.X_train.shape[1]))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(28 * 28))
        model.add(Reshape((28, 28, 1)))
        model.add(Conv2D(32, ((7, 7)), activation = 'sigmoid'))
        model.add(Conv2D(32, ((7, 7)), activation='sigmoid'))
        model.add(Flatten())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        img = Input(shape=(self.X_train.shape[1], ))
        validity = model(img)

        return Model(img, validity)

    def build_discriminator_coo(self):

        model = Sequential()
        # model.add(Flatten(input_shape=self.X_train.shape[1]))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(28 * 28))
        model.add(Reshape((28, 28, 1)))
        model.add(Conv2D(32, ((7, 7)), activation = 'sigmoid'))
        model.add(Conv2D(32, ((7, 7)), activation='sigmoid'))
        model.add(Flatten())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        img = Input(shape=(self.X_train.shape[1], ))
        validity = model(img)

        return Model(img, validity)

    def judge_overlap(self, img_fake, check_batch = 10):
        self.judge_count += 1
        check = self.discriminator_coo.predict(img_fake)
        score_mean = np.mean(check.reshape((1, -1)))
        print('score_mean:', score_mean, 'self.judge_count % check_batch:','self.score_pre:', self.score_pre)
              # self.judge_count % check_batch, 'np.mean(self.score_mean_list):',
              # np.mean(self.score_mean_list))

        # self.score_mean_list.append(score_mean)
        # if len(self.score_mean_list) >= 40:
        #     self.score_mean_list.pop(0)
        if self.judge_count % check_batch == 0:  #判断是否回滚
            if score_mean >= self.score_pre:
                self.score_pre = score_mean
                return True
            else:
                self.score_pre = score_mean
                return False
        else:
            self.score_pre = score_mean
            return False


        # self.generator.predict(img_fake)

        # if loss1 <= 0.0021386:
        #     return False
        # self.judge_count += 1
        # self.loss1_list.append(loss1)
        # if self.judge_count % check_batch == 0:
        #     if loss1 >= np.mean(self.loss1_list):
        #         return True
        #     else:
        #         return False
        # else:
        #     return False




    def train(self, epochs, batch_size=128, sample_interval=20):

        global loss_gancoo

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        X_train = self.X_train
        X_side = self.X_side


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if self.is_coo:
            epoch_Dadv = 5
        else:
            epoch_Dadv = 2


        for epoch in range(epochs):

            for __ in range(epoch_Dadv):


                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator
                # self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                if self.is_coo:
                    idx = np.random.randint(0, X_side.shape[0], batch_size)
                    imgs_side = X_side[idx]
                    # print('gen_imgs_1:', gen_imgs.shape)
                    d_loss_real_coo = self.discriminator_coo.train_on_batch(imgs_side, valid)
                    d_loss_fake_coo = self.discriminator_coo.train_on_batch(gen_imgs, fake)


            # ---------------------
            #  Train Coo
            # ---------------------

            epoch_coo = 50

            # for _ in range(batch_size):
            #     self.discriminator.trainable = False
            #     if _ % epoch_coo != 0:
            #         continue
            check_batch = 1


            if self.is_coo:
                # self.discriminator.trainable = False
                # print('loss_gancoo:', loss_gancoo)
                # idx = np.random.randint(0, X_side.shape[0], epoch_coo)
                # imgs_side = X_side[idx]
                noise = np.random.normal(0, 1, (epoch_coo, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                # print('gen_imgs_2:', gen_imgs.shape)
                valid_ = np.ones((epoch_coo, 1))
                fake_ = np.zeros((epoch_coo, 1))

                # d_loss_real = self.discriminator_coo.train_on_batch(imgs_side, valid_)
                # d_loss_fake = self.discriminator_coo.train_on_batch(gen_imgs, fake_)
                # if self.judge_overlap(d_loss_real[0], d_loss_fake[0], check_batch = check_batch) and _ != 0:#回滚
                if self.judge_overlap(gen_imgs, check_batch=check_batch) and epoch != 0:
                    print("rollback!!!")
                    if random() <= 0.1:

                        lr = K.get_value(self.combined.optimizer.lr)
                        print('lr:', lr)
                        K.set_value(self.combined.optimizer.lr, lr * 0.99)

                    denine_epoches = epoch % check_batch + 1
                    # self.generator = self.generator_pre
                    # d_loss_real = self.discriminator.train_on_batch(imgs_side, fake_)
                    pre_weights = self.this_weights
                    this_weights = np.array(self.generator.get_weights())
                    #回滚
                    new_weights = pre_weights + (this_weights - pre_weights) * 0.9  #0.9可调试
                    self.generator.set_weights(new_weights)

                    noise = np.random.normal(0, 1, (denine_epoches, self.latent_dim))
                    idx = np.random.randint(0, X_side.shape[0], denine_epoches)
                    imgs_side = X_side[idx]
                    gen_imgs = self.generator.predict(noise)
                    # print('gen_imgs_1:', gen_imgs.shape)
                    valid_overlap = np.ones((denine_epoches, 1))
                    fake_overlap = np.zeros((denine_epoches, 1))
                    d_loss_real_coo = self.discriminator_coo.train_on_batch(imgs_side, valid_overlap)
                    d_loss_fake_coo = self.discriminator_coo.train_on_batch(gen_imgs, fake_overlap)
                    # valid_overlap = np.ones((denine_epoches, 1))
                    # g_loss_overlap = self.combined.train_on_batch(noise, valid_overlap)
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    # self.combined_coo.train_on_batch(noise, fake)



                    # continue
                    # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)


                #获取loss
                # noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # gen_imgs = self.generator.predict(noise)
                # gan_coo_pre = self.discriminator_coo.predict(gen_imgs)
                # loss_gancoo = losses.binary_crossentropy(1, gan_coo_pre)

                # loss_gancoo = d_loss_real[0]
                # for ___ in range(3):

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)

                    # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                print('d_loss_real:', d_loss_real_coo, ' d_loss_fake:', d_loss_fake_coo, ' g_loss:', g_loss)

                # If at save interval => save generated image samples
                # if epoch % sample_interval == 0:
                #     self.sample_images(epoch)
                if epoch % check_batch == 0:
                    self.this_weights = np.array(self.generator.get_weights())


            # ---------------------
            #  Train Generator
            # ---------------------
            else:
                # self.discriminator.trainable = False

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, valid)

                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                print('d_loss_real:', d_loss_real, ' d_loss_fake:', d_loss_fake, ' g_loss:', g_loss)


                    # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        num_return = self.X_train.shape[0]
        noise = np.random.normal(0, 1, (num_return, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        print(gen_imgs)
        if random() <= 0.5:
            try:
                plt.imshow(gen_imgs[0].reshape((28, 28)), cmap='gray')
                plt.show()
            except:
                pass
        self.GAN_process.append(gen_imgs)

        plt.cla()



# def loss_for_GAN(y_true, y_pred_GAN):
#     global loss_gancoo
#     print('loss_gancoo:', loss_gancoo)
#     loss_return = losses.binary_crossentropy(y_true, y_pred_GAN) - 2 * loss_gancoo
#     return loss_return

loss_gancoo = 0
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)