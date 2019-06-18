import numpy as np


def gconvbias(bias, output):
    bias.grad += np.sum(output.grad)
    return bias


def gconvfilter(filter, input, output, stride):
    s = stride
    (xfilt, yfilt, zfilt, nfilt) = np.shape(filter.val)
    (xout, yout, nout) = np.shape(output.val)
    for a in range(xfilt):
        for b in range(yfilt):
            for c in range(zfilt):
                for n in range(nfilt):
                    for i in range(xout):
                        for j in range(yout):
                            filter.grad[a, b, c, n] += input.val[a + s * i, b + s * j, c] * output.grad[i, j, n]
    return filter


def gconvinput(input, filter, output, stride):
    input.grad = np.zeros_like(input.val)
    s = stride
    (xfilt, yfilt, zfilt, nfilt) = np.shape(filter.val)
    (xout, yout, nout) = np.shape(output.val)
    for a in range(xfilt):
        for b in range(yfilt):
            for c in range(zfilt):
                for n in range(nfilt):
                    for i in range(xout):
                        for j in range(yout):
                            input.grad[a + s * i, b + s * j, c] += filter.val[a, b, c, n] * output.grad[i, j, n]
    return input


def grelu(input, output):
    input.grad += output.grad * np.where(input.val > 0, 1, 0)
    return input


def gmaxpool(input, d, output):
    input.grad += np.zeros_like(input.val)
    (xout, yout, zout) = np.shape(output.val)
    for i in range(xout):
        for j in range(yout):
            for k in range(zout):
                for a in range(d):
                    for b in range(d):
                        if output.val[i, j, k] == input.val[d * i + a, d * j + b, k]:
                            input.grad[d * i + a, d * j + b, k] = output.grad[i, j, k]
    return input


def gFCinput(input, weight, output):
    (xw,yw,nw,sw) = np.shape(weight.val)
    for s in range(sw):
        input.grad += output.grad[s] * weight.val[:,:,:,s]
    return input


def gFCweight(weight, input, output):
    (xout) = np.size(output.val)
    (xi,yi,ni) = np.shape(input.val)
    for s in range(xout):
        for n in range(ni):
            weight.grad[:,:,n,s] += output.grad[s] * input.val[:,:,n]
    return weight


def gFCbias(bias, output):
    bias.grad += output.grad
    return bias

def gscores(scores, softmax, true):
    scores.grad += ((-1 * true) + 1) * softmax.val + true * (softmax.val - 1)
    return scores

def gpad(input,pimg,pad):
    (xin,yin,_)= np.shape(pimg.val)
    input.grad += pimg.grad[pad:xin-pad,pad:yin-pad,:]
    return input



