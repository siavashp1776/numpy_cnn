import run_CNN as c
import numpy as np
import copy
import batchSGD as SGD
import pprint

vars = c.initialize_vars()
CIFAR_img = np.random.rand(32,32,3)
CIFAR_img = CIFAR_img-np.average(CIFAR_img)

[vars,correct,loss] = c.run_CNN(CIFAR_img,0,vars)

prevars = copy.deepcopy(vars)

var_stats = {}

e_g2 = {}
e_x2 = {}

p = np.float(0.95)
e = np.float(1e-6)

for var,i in zip(vars,list(range(np.size(vars)))):
    e_g2.update({i:np.zeros_like(var.val)})
    e_x2.update({i:np.zeros_like(var.val)})


p = np.float(0.95)
e = np.float(1e-6)

pp = pprint.PrettyPrinter(indent=4)

[vars,e_g2,e_x2] = SGD.adaDeltaUpdate(vars,p,e,e_g2,e_x2)

for var,i in zip(vars,list(range(np.size(vars)))):
    var_name = {0: 'cfilt1', 1: 'cbias1', 2: 'cfilt2', 3: 'cbias2', 4: 'cfilt3', 5: 'cbias3', 6: 'FCweights',
                7: 'FCbias'}
    var_stats[var_name[i]] = np.array([[np.average(var.val), np.std(var.val), np.average(var.grad), np.std(var.grad),0,0]])






