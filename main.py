import numpy as np
import openCIFAR as oC
import feedforward as f
import backprop as b
import matplotlib.pyplot as plt
import batchSGD as SGD
import run_CNN as c
import pprint

## SET UP BATCH SGD PARAMETERS
len = int(10e3) # for CIFAR 10 only
batch_size = 4 #yeah IDK man
alpha = 0.001

iterations = 250
acc = []
accArr = []
lossArr = []

vars = c.initialize_vars()
e_g2 = {}
e_x2 = {}

for var,i in zip(vars,list(range(np.size(vars)))):
    e_g2.update({i:np.zeros_like(var.val)})
    e_x2.update({i:np.zeros_like(var.val)})

p = np.float(0.99)
e = np.float(1e-6)

pp = pprint.PrettyPrinter(indent=4)

## SELECT AND RUN BATCH
for iter in range(int(iterations)):
    [vars,correct, loss] = SGD.select_and_run_batch(len,batch_size,vars)

    var_stats = {}
    i = 0
    for var in vars:
        var_name = {0:'cfilt1', 1:'cbias1', 2:'cfilt2', 3:'cbias2', 4:'FCweights', 5:'FCbias'}
        var_stats.update({var_name[i]:[np.average(var.val),np.std(var.val),np.average(var.grad),np.std(var.grad)]})
        i+= 1

    pp.pprint(var_stats)
    print('Loss:',loss)
    print('Correct?  ', correct)
    lossArr.extend(loss)
    acc.extend(correct)
    for i in range(iter,iter+batch_size):
        accArr.append(np.sum(acc[iter:iter+i])/(iter+i+1))
    print('Accuracy:  ',np.sum(acc)/np.size(acc))
    print('Num examples: ', np.size(acc))
    print('')

    # SGD.weight_update(vars,float(alpha)/(iter+1))
    [vars,e_g2,e_x2] = SGD.adaDeltaUpdate(vars,p,e,e_g2,e_x2)
    if iter%10 == 0:
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        x = [i for i in range(np.size(acc))]
        axes.plot(x, accArr)
        fig.savefig('Acc.png')
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        x = [i for i in range(np.size(lossArr))]
        axes.plot(x, lossArr)
        fig.savefig('Loss.png')




# plt.show()
#
# ## FIRST OPEN CIFAR
# (a, label) = oC.plot_CIFAR(5)
# true = np.zeros((1, 10))
# true[0, label] = 1
#
# ## THEN DEFINE OBJECT CLASS
#
# class var:
#     val = []
#     grad = []
#
#
# ## THEN INITIALIZE VARIABLES! WERE READY TO CNN NOW!
# input = var()
# input.val = a
# input.grad = np.zeros_like(input.val)
#
# pimg = var()
#
# filter = var()
# filter.val = np.random.rand(5, 5, 3)
# filter.grad = np.zeros_like(filter.val)
#
# bias = var()
# bias.val = -25 * np.random.rand()
# bias.grad = 0
#
# pad = 2
# stride = 1
# d = 2  ## receptive field for non overlapping maxpool
#
# ## FIRST PAD
#
# pimg = f.pad_image(input, pad)
#
# ## THEN CONVOLVE
# convolved = f.convolve(pimg, filter, bias, stride)
# fig = plt.figure(figsize=(3, 3))
# ax = fig.add_subplot(111)
# ax.imshow(convolved.val, interpolation='bicubic')
# ax.set_title('Convolved image')
# plt.show()
#
# ## THEN RELU
# RELUd = f.relu(convolved)
# fig = plt.figure(figsize=(3, 3))
# ax = fig.add_subplot(111)
# ax.imshow(RELUd.val, interpolation='bicubic')
# ax.set_title('Relu')
# plt.show()
#
# ## THEN MAXPOOL
#
# maxpooled = f.maxpool(RELUd, d)
# fig = plt.figure(figsize=(3, 3))
# ax = fig.add_subplot(111)
# ax.imshow(maxpooled.val, interpolation='bicubic')
# ax.set_title('Maxpool')
# plt.show()
#
# ## THEN DEFINE MORE VARIABLES BECAUSE MAX POOL MAKES THINGS SMALLER
# (xr, yr) = np.shape(maxpooled.val)
# FCbias = var()
# FCbias.val = np.random.rand(10)
# FCbias.grad = np.zeros_like(FCbias.val)
# FCweights = var()
# FCweights.val = np.random.rand(xr, yr, np.size(FCbias.val)) / 100
# FCweights.grad = np.zeros_like(FCweights.val)
#
# ## THEN FULLY CONNECT
# score = f.fullyconnected(maxpooled, FCweights, FCbias, (1, 10))
# print(score.val)  ## scores
#
# ## THEN SOFTMAX LOSS
# score.val -= np.max(score.val)
# softmax = var()
# softmax.val = np.exp(score.val) / np.sum(np.exp(score.val))
# loss = var()
# loss.grad = 1
# loss.val = f.CLLoss(softmax.val, true)
#
# print(softmax.val)
# print(loss.val)
# print('yey')
#
#
# ## THEN BACKPROP (JUST THE CONVOLUTION HERE)
#
# score = b.gscores(score, softmax, true)
#
# FCweights = b.gFCweight(FCweights, maxpooled, score)
# FCbias = b.gFCbias(FCbias, score)
# maxpooled = b.gFCinput(maxpooled, FCweights, score)
#
# RELUd = b.gmaxpool(RELUd, d, maxpooled)
#
# convolved = b.grelu(convolved, RELUd)
#
# bias = b.gbias(bias, convolved)
# filter = b.gfilter(filter, pimg, convolved, stride)
# pimg = b.ginput(pimg, filter, convolved, stride)
#
# vars = [FCweights,FCbias,bias,filter]
#
# print('yeee')
