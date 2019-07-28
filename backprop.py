import numpy as np


def gconvbias(bias, output):
    """
    updates gradients of bias tensor used for convolution
    e.g.:
        bias = b.gconvbias(bias,conv1)

    :param bias: tensor of biases used for convolution
    :param output: output of convolution
    :return: bias parameter, with updated gradients
    """
    bias.grad += np.sum(output.grad,(0,1))
    return bias


def gconvfilter(filter, input, output, stride):
    """
    updates gradients of convolution filters used for convolution
    e.g.:
        cfilt1 = b.gconvfilter(cfilt1, pimg, conv1, stride)

    :param filter: convolution filter, e.g. cfilt1
    :param input: input to convolution, e.g. pimg
    :param output: output of convolution, e.g. conv1
    :param stride: stride used in convolution,
    :return: convolution filter, with updated gradients, e.g cfilt1
    """
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
    """
    updates gradients of the tensor that was an input to a forward convoultion
    e.g.
        pimg = b.gconvinput(pimg,cfilt1,conv1,stride)


    :param input: input of convolution operation, e.g. pimg
    :param filter: convolution filter, e.g. cfilt1
    :param output: output of convolution, e.g. conv1
    :param stride: stride used for convolution
    :return: input of convolution with updated gradients. e.g. pimg
    """
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
    """
    Updates gradients of a tensor that is passed into a relu operation
    e.g.
        conv1 = g.grelu(conv1,relu1)
    :param input: input into relu operation, e.g. conv1
    :param output: output of relu operation, e.g. relu1
    :return: output of relu operation with updated gradients, e.g. relu1
    """
    input.grad += output.grad * np.where(input.val > 0, 1, 0)
    return input


def gmaxpool(input, d, output):
    """
    Updates gradients of a tensor that is passed into a maxpool operation
    e.g.
        relu1 = b.gmaxpool(relu1,d,maxpooled1)

    :param input: input of maxpool operation, e.g. relu1
    :param d: receptive field of maxpool operation
    :param output: output of maxpool operation, e.g. maxpooled1
    :return: input of maxpool operation with updated gradients, e.g. relu1
    """
    input.grad += np.zeros_like(input.val)
    (xout, yout, zout) = np.shape(output.val)
    for i in range(xout):
        for j in range(yout):
            for k in range(zout):
                for a in range(d):
                    for b in range(d):
                        if output.val[i, j, k] == input.val[d * i + a, d * j + b, k]:
                            input.grad[d * i + a, d * j + b, k] += output.grad[i, j, k]
    return input


def gFCinput(input, weight, output):
    """
    Updates gradients of a tensor that is an input into a FC layer
    e.g.:
        maxpooled3 = b.gFCinput(maxpooled3, FCweights, scores)
    :param input: input into FC layer, e.g. maxpooled3
    :param weight: FC weights, e.g. FCweights
    :param output: output of FC layer, e.g. scores
    :return: input into FC layer with updated gradients, e.g. maxpooled3
    """
    (xw,yw,nw,sw) = np.shape(weight.val)
    for s in range(sw):
        input.grad += output.grad[s] * weight.val[:,:,:,s]
    return input


def gFCweight(weight, input, output):
    """
    Updates gradients of weight tensor used in a fully connected layer
    e.g.:
        FCweights = b.gFCweight(FCweights,maxpooled3,scores)
    :param weight: weight tensor used in a fully connected layer, e.g. FCweights
    :param input: input into FC layer, e.g. maxpooled3
    :param output: output of FC layer, e.g. scores
    :return: weight tensor used in a fully connected layer with updated gradients, e.g. FCweights
    """
    (xout) = np.size(output.val)
    (xi,yi,ni) = np.shape(input.val)
    for s in range(xout):
        for n in range(ni):
            weight.grad[:,:,n,s] += output.grad[s] * input.val[:,:,n]
    return weight


def gFCbias(bias, output):
    """
    Updates gradients of bias tensor of an FC layer
    e.g.:
        FCbias = b.gFCbias(FCbias,scores)

    :param bias: bias tensor used in FC layer, e.g. FCbias
    :param output: output of FC operation, e.g. scores
    :return: bias tensor with updated weights, e.g. FCbias
    """
    bias.grad += output.grad
    return bias

def gscores(scores, softmax, true):
    """
    Calculates gradients of a score tensor
    e.g.:
        scores = b.gscores(scores, softmax, true)

    :param scores: score array computed as final output, before softmax/cross-entropy loss, e.g. scores
    :param softmax: values of softmax function, e.g. softmax
    :param true: array of truths (binary array), e.g. true
    :return: score array with updated gradients, e.g. scores
    """
    scores.grad += ((-1 * true) + 1) * softmax.val + true * (softmax.val - 1)
    return scores

def gpad(input,pimg,pad):
    """
    Calculates gradients of an image that is passed into the pad function
    e.g.
        maxpooled1 = b.gpad(maxpooled1,pimg2,pad)

    :param input: input in to the pad function, e.g. maxpooled1
    :param pimg: padded image, e.g. pimg2
    :param pad: padding amount
    :return: updates gradients of a tensor passed into the pad function, e.g. maxpooled1
    """
    (xin,yin,_)= np.shape(pimg.val)
    input.grad += pimg.grad[pad:xin-pad,pad:yin-pad,:]
    return input