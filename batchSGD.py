import numpy as np
import openCIFAR as oC
import run_CNN as c

def select_and_run_batch(len,batch_size,vars):
    batch = np.random.choice(range(len),batch_size,False)
    # print('yoo')
    carr = []
    larr = []
    for i in batch:
        (img,label) = oC.plot_CIFAR(i)
        [vars, correct, loss] = c.run_CNN(img,label,vars)
        carr.append(correct)
        larr.append(loss.val)
    # print('yeet')

    return [vars, carr, larr] ### stochastic with the last elem in batch

def weight_update(vars,alpha,momentum):
    for var in vars:
        var.val -= alpha* var.grad
    vars = c.reinitgrads(vars)
    return vars

# weight_update(vars,alpha)

def adaDeltaUpdate(vars,p,epsilon,e_g2,e_x2):
    for var,i in zip(vars,list(range(np.size(vars)))):
        e_g2[i] = p*e_g2[i]+(1-p)*np.power(var.grad,2)
        rms_gt =  np.sqrt(e_g2[i]+epsilon)
        rms_delta_x = np.sqrt(e_x2[i]+epsilon)
        delta_x = -rms_delta_x*var.grad/rms_gt
        e_x2[i] = p*e_x2[i]+(1-p)*np.power(delta_x,2)
        var.val+=delta_x
    return [vars,e_g2,e_x2]


