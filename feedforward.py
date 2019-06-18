import numpy as np
# import matplotlib.pyplot as plt
# import openCIFAR


class var:
    val = []
    grad = []

#
# a = openCIFAR.plot_CIFAR(3)
#
# # YEEHAW VARIABLE INITIALIZATION
# input = var()
# input.val = a
# input.grad = np.zeros_like(input.val)
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

#
# #
# convolved = convolve(input, filter, bias, pad, stride)
# fig = plt.figure(figsize=(3, 3))
# ax = fig.add_subplot(111)
# ax.imshow(convolved.val, interpolation='bicubic')
# ax.set_title('Convolved image')
# plt.show()


## relu

def relu(input):
    """
    :param input:
    :return:
    """
    ans = var()
    ans.val = input.val * (input.val > 0)
    ans.grad = np.zeros_like(ans.val)
    return ans


# RELUd = relu(convolved)
#
# fig = plt.figure(figsize=(3, 3))
# ax = fig.add_subplot(111)
# ax.imshow(RELUd.val, interpolation='bicubic')
# ax.set_title('Relu')
# plt.show()


## maxpool

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

#
# maxpooled = maxpool(RELUd, 2)
# fig = plt.figure(figsize=(3, 3))
# ax = fig.add_subplot(111)
# ax.imshow(maxpooled.val, interpolation='bicubic')
# ax.set_title('Maxpool')
# plt.show()


## fully connected
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

# (xr, yr) = np.shape(maxpooled.val)
# b = var()
# b.val = np.random.rand(10)
# FCweights = var()
# FCweights.val = np.random.rand(xr, yr, np.size(b.val)) / 100
# score = fullyconnected(maxpooled, FCweights, b, (1, 10))
# print(score.val)
#
# # ## softmax
# nj = score.val / np.sum(score.val)
# soft2 = np.exp(nj) / np.sum(np.exp(nj))
# soft = np.exp(score.val) / np.sum(np.exp(score.val))
#
# corr = np.zeros((1, 10))
# corr[0, 4] = 1

def CLLoss(soft, corr):
    return np.sum(-corr * np.log(soft))

# print(soft)
# print(soft2)
# print(CLLoss(soft, corr))
# print(CLLoss(soft2, corr))
# print('yey')
