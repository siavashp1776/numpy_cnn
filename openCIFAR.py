import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

plt.close('all')


def unpickle(file):
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


pd_tr = pd.DataFrame()
tr_y = pd.DataFrame()

for i in range(1, 6):
    data = unpickle('data_batch_' + str(i))
    pd_tr = pd_tr.append(pd.DataFrame(data[b'data']))
    tr_y = tr_y.append(pd.DataFrame(data[b'labels']))
    pd_tr['labels'] = tr_y

tr_x = np.asarray(pd_tr.iloc[:, :3072])
tr_y = np.asarray(pd_tr['labels'])
ts_x = np.asarray(unpickle('test_batch')[b'data'])
ts_y = np.asarray(unpickle('test_batch')[b'labels'])
labels = unpickle('batches.meta')[b'label_names']


def plot_CIFAR(ind):
    arr = tr_x[ind]
    R = arr[0:1024].reshape(32, 32) / 255.0
    G = arr[1024:2048].reshape(32, 32) / 255.0
    B = arr[2048:].reshape(32, 32) / 255.0
    k = 0.4729027662888078 ### GLOBAL PIXEL AVERAGE, NOT PER CHANNEL
    img = np.dstack((R, G, B))-k
    # title = re.sub('[!@#$b]', '', str(labels[tr_y[ind]]))
    # fig = plt.figure(figsize=(3, 3))
    # ax = fig.add_subplot(111)
    # ax.imshow(img, interpolation='bicubic')
    # ax.set_title('Category = ' + title, fontsize=15)
    # plt.show()
    return (img,tr_y[ind])