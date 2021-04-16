#字典按照values排序，返回排好序x_list和y_list
import numpy as np
def sort_by_values(dic):
    xlist = []
    ylist = []
    dic_sorted = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    for w in dic_sorted:
        #f1.writelines(str(w[0]) + "s:" + str((w[1] / count_total) * 1000) + "‰" + "\n")
        xlist.append(w[0])
        ylist.append(w[1])
    return np.array(xlist),np.array(ylist)

def sort_by_keys(dic):
    xlist = []
    ylist = []
    dic_sorted = sorted(dic.items(), key=lambda x: x[0], reverse=False)
    for w in dic_sorted:
        #f1.writelines(str(w[0]) + "s:" + str((w[1] / count_total) * 1000) + "‰" + "\n")
        xlist.append(w[0])
        ylist.append(w[1])
    return np.array(xlist),np.array(ylist)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


def versiontuple(v):
    return tuple(map(int, (v.split("."))))

#画图和分类，参数：（样本参数n维数组，样本类标，模型对象， ）
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

#合并两个特征成一个n*2维的数组
def merging_characteristics(x,y):
    return np.hstack((x.reshape(len(x),1),y.reshape(len(y),1)))
