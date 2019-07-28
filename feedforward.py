import numpy as np


class var:
    val = []
    grad = []

def pad_image(input, pad):
    """
    :param input: input image object (values and gradients)
    :param pad: number of pixels to pad on each side
    :return: padded image object
    """
    output = var()
    output.val = np.pad(input.val, ((pad, pad), (pad, pad), (0, 0)), 'constant')
    output.grad = np.zeros_like(output.val)
    return output

def convolve(input, filter, bias, stride):
    """
    input = ALREADY PADDED IMAGE (32 * 32 * 3) OR TENSOR of previous activations (x*y*n)
    filter = TENSOR OF MULTIPLE convolution filters. typically 5x5x3xm, or 5x5x16xm
    bias = Array of values (m) to add to filter after multiplying with input image
    pad = number of pixels to pad in x and y directions
    stride = number of pixels filter should skip during convolution
    """
    res = var()
    (xpimg, ypimg, zpimg) = np.shape(input.val)
    (xfilt, yfilt, zfilt,nfilt) = np.shape(filter.val)
    w = int((xpimg - xfilt) / stride + 1)
    h = int((ypimg - yfilt) / stride + 1)
    res.val = np.zeros([w, h, nfilt])
    for i in range(w):
        for j in range(h):
            for n in range(nfilt):
                area = input.val[j:xfilt + j, i:yfilt + i, :]
                res.val[j, i, n] = np.sum(area * filter.val[:,:,:,n]) + bias.val[n]
    res.grad = np.zeros_like(res.val)
    return res


def relu(input):
    """
    :param input:
    :return:
    """
    ans = var()
    ans.val = input.val * (input.val > 0)
    ans.grad = np.zeros_like(ans.val)
    return ans


def maxpool(input, d):
    """
    performs maxpool operation on an input image object
    :param input: input image object
    :param d: dimension of maxpool receptive field. typically 2
    :return: returns an image object, with values = maxpooled from input
    """
    result = var()
    (x, y, z) = np.shape(input.val)
    result.val = np.zeros([int(x / d), int(y / d), int(z)])
    for i in range(int(x / d)):
        for j in range(int(y / d)):
            for k in range(z):
                result.val[i, j, k] = np.max(input.val[d * i:d * (i + 1), d * j:d * (j + 1), k])
    result.grad = np.zeros_like(result.val)
    return result


def fullyconnected(input, w, b, outputsize):
    scores = var()
    scores.val = np.zeros(outputsize)  # each entry of the output
    (xr, yr, zr) = np.shape(input.val)
    for s in range(np.size(scores.val)):
        for i in range(xr):
            for j in range(yr):
                for k in range(zr):
                    scores.val[s] += w.val[i, j, k, s] * input.val[i, j, k]
        scores.val[s] += b.val[s]
    scores.grad = np.zeros_like(scores.val)
    return scores


def CLLoss(soft, corr):
    return np.sum(-corr * np.log(soft))