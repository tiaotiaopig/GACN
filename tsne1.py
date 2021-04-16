#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
from keras2 import *
import numpy as np
import pylab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from simple_keras import *
import random
from collections import Counter

def GACN(X, y, coo = True):
    X_aug = np.zeros(X.shape[1])
    y_aug = [0]
    X_aug_pure = np.zeros(X.shape[1])
    y_aug_pure = [0]

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
            X_aug_pure = np.vstack(((X_aug_pure, X_intra_small)))
            y_aug_pure = y_aug_pure + X_intra_small.shape[0] * [label, ]
            continue
        print(label)
        #side
        X_side = X[labels_for_col != label, :]
        # X_side = X[labels_for_col == 'BENIGN', ]
        # gan, gan_coo, generator, discriminator, discriminator_coo = build_model(X_intra_small.shape[1], rad = 300)

        if coo:
            gan = GAN(X_train=X_intra_small, X_side=X_side)

        else:#compare
            gan = GAN(X_train=X_intra_small, X_side=None)

        gan.train(epochs=1000, batch_size=10, sample_interval=10)
        generator = gan.generator
        noise = gan.return_random_noise(sample_num = 400)
        #生成X放入Aug
        X_fake = generator.predict(noise)
        X_aug = np.vstack((X_aug, X_fake))
        #单独fake
        X_aug_pure = np.vstack(((X_aug_pure, X_fake)))
        y_aug_pure = y_aug_pure + X_fake.shape[0] * [label + '_fake', ]
        #y
        y_aug = y_aug + X_fake.shape[0] * [label + '_fake', ]
    y_aug = np.array(y_aug)
    y_aug_pure = np.array(y_aug_pure)
    return X_aug[1:,], y_aug[1:], X_aug_pure[1:,], y_aug_pure[1:],

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if sumP == 0:
        return 0, 0


    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        if type(thisP) == type(np.array([])):
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
        else:
            print("Zreo value!!")

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))

    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=70, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 300
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 200
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y

def sample_B(l):
    if l == 'BENIGN' and random.random() <= 0.5:
        return True
    elif random.random() <= 0.2:
        return True
    else:
        return False

def sample_B_plt(l):
    # if random.random() <= 0.2:
    #     return True
    # else:
    #     return False
    if l == 'BENIGN':
        if random.random() <= 0.16:
            return True
        else:
            return False
    else:
        return True

#
if __name__ == "__main__":
    # print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")
    # X = np.loadtxt("mnist2500_X.txt")
    df = pd.read_csv(r"D:\Experi\feature_vec_total_small.csv")
    df = df.iloc[1:, :]
    label_list_ori = np.array(df.iloc[:, -1])

    df = df[list(map(sample_B, label_list_ori))]

    X = np.array(df.iloc[:, :-1])
    y_ori = np.array(df.iloc[:, -1])
    print(Counter(y_ori))

    # sample_index = np.random.randint(0, X.shape[0], 1000)
    # X = X[sample_index]
    # label_list_ori = label_list_ori[sample_index]

    X_aug, y_aug, X_aug_pure, y_aug_pure = GACN(X, y_ori, coo= False)

    X_aug = X_aug_pure
    y_aug = y_aug_pure

    #对Bengin进行删减
    index_delete_for_plt = list(map(sample_B_plt, y_aug))
    X_aug = X_aug[index_delete_for_plt]
    y_aug = y_aug[index_delete_for_plt]
    print(Counter(y_aug))




    # lab = LabelEncoder()
    # label_list_ori = lab.fit_transform(label_list_ori)
    # print(lab.inverse_transform([1]))
    # print(label_list_ori)

    # labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X_aug, 2, 50, 20.0)
    # color_list = ['r', 'y', 'b', 'g', 'purple']
    # color_dic = {'DoS attacks-SlowHTTPTest': 'r', 'DoS attacks-GoldenEye': 'y', 'DDOS attack-HOIC':'g', 'Infilteration' : 'purple',
    #              'DoS attacks-Slowloris': 'orange','SSH-Bruteforce':'#FFC0CB', 'DoS attacks-Hulk':'#4169E1', 'FTP-BruteForce': '#2E8B57',
    #              'DDOS attack-LOIC-UDP' : '#9ACD32'}
    color_dic = {'DoS slowloris': 'r', 'DoS Slowhttptest': 'y', 'DoS Hulk':'g', 'DoS GoldenEye' : 'purple'}
    color_choice = 0
    plted_list = []
    for label_ in set(y_aug):
        if label_ in plted_list:
            continue
        if label_[-5:] == '_fake':
            label_true = label_[:-5]
            label_fake = label_
            print('label_fake:', label_true)
            print('label_true:', label_true)
            try:
                color = color_dic[label_true]
            except:
                continue

            X_plt_fake = Y[y_aug == label_fake]
            X_plt_true = Y[y_aug == label_true]

            plt.scatter(X_plt_fake[:, 0], X_plt_fake[:, 1], label=label_true, s=10,alpha=0.7, c = color, marker = 'x')
            # plt.scatter(X_plt_true[:, 0], X_plt_true[:, 1], label=label_true, s=10,alpha=0.7, c = color, )
            plted_list.append(label_true)
            plted_list.append(label_fake)
            color_choice += 1
        elif label_ == 'BENIGN':#benign
            print('BENIGN:', label_)

            X_plt_true = Y[y_aug == label_]
            plt.scatter(X_plt_true[:, 0], X_plt_true[:, 1], label=label_, s=25, alpha=0.4,
                        c='b', )
            color_choice += 1


    plt.legend()
    plt.show()


    # pylab.scatter(Y[:, 0], Y[:, 1], 20, label_list_ori)
    # plt.legend()
    # pylab.show()
