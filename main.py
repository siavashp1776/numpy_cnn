import numpy as np
import openCIFAR as oC
import feedforward as f
import backprop as b
import matplotlib.pyplot as plt
import batchSGD as SGD
import run_CNN as c
import pprint
import copy

## SET UP BATCH SGD PARAMETERS
len = int(10e3) # for CIFAR 10 only
batch_size = 4 #yeah IDK man
alpha = 0.002

iterations = 250
acc = []
accArr = []
lossArr = []

vars = c.initialize_vars()
e_g2 = {}
e_x2 = {}

var_stats = {}

for var,i in zip(vars,list(range(np.size(vars)))):
    var_name = {0: 'cfilt1', 1: 'cbias1', 2: 'cfilt2', 3: 'cbias2', 4: 'cfilt3', 5: 'cbias3', 6: 'FCweights',
                7: 'FCbias'}
    var_stats[var_name[i]] = np.array([[np.average(var.val), np.std(var.val), np.average(var.grad), np.std(var.grad),0,0]])

for var,i in zip(vars,list(range(np.size(vars)))):
    e_g2.update({i:np.zeros_like(var.val)})
    e_x2.update({i:np.zeros_like(var.val)})


p = np.float(0.95)
e = np.float(1e-6)

pp = pprint.PrettyPrinter(indent=4)

## SELECT AND RUN BATCH
for iter in range(int(iterations)):
    vars = c.reinitgrads(vars)
    prevars = copy.deepcopy(vars)
    [vars,correct, loss] = SGD.select_and_run_batch(len,batch_size,vars)

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

    #vars = SGD.weight_update(vars,float(alpha))
    [vars,e_g2,e_x2] = SGD.adaDeltaUpdate(vars,p,e,e_g2,e_x2)

    for var, i in zip(vars, list(range(np.size(vars)))):
        var_name = {0: 'cfilt1', 1: 'cbias1', 2: 'cfilt2', 3: 'cbias2', 4: 'cfilt3', 5: 'cbias3', 6: 'FCweights',
                    7: 'FCbias'}
        var_stats[var_name[i]] = np.concatenate((var_stats[var_name[i]], np.array(
            [[np.average(var.val), np.std(var.val), np.average(var.grad), np.std(var.grad),np.linalg.norm(prevars[i].val-var.val),np.linalg.norm(prevars[i].grad-var.grad)]])))

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