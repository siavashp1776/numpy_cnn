import numpy as np
import openCIFAR as oC
import feedforward as f
import backprop as b
import matplotlib.pyplot as plt

## FIRST INITIALIZE ALL VARIABLES

def initialize_vars():
    class var:
        val = []
        grad = []

    cfilt1 = var()
    cfilt1.val = np.random.rand(5, 5, 3, 16)*np.sqrt(2/(32*32*3)) # Ximage, Y image, Z image, n filters applied
    cfilt1.grad = np.zeros_like(cfilt1.val)

    cbias1 = var()
    cbias1.val = 1e-5+np.random.rand(16)*np.sqrt(2/(32*32*3)) ## filters
    cbias1.grad = np.zeros_like(cbias1.val)

    cfilt2 = var()
    cfilt2.val = np.random.rand(5, 5, 16, 20)*np.sqrt(2/(32*32*16)) #X input, Y input, Z channels in (16), N filters out
    cfilt2.grad = np.zeros_like(cfilt2.val)

    cbias2 = var()
    cbias2.val = 1e-5+np.random.rand(20)*np.sqrt(2/(32*32*16)) ## filters
    cbias2.grad = np.zeros_like(cbias2.val)

    FCweights = var()
    FCweights.val = np.random.rand(8, 8, 20, 10)*np.sqrt(2/(32*32*20)) ## x, y, n filters, scores
    FCweights.grad = np.zeros_like(FCweights.val)

    FCbias = var()
    FCbias.val = 1e-5+np.random.rand(10)*np.sqrt(2/(32*32*20)) ## scores
    FCbias.grad = np.zeros_like(FCbias.val)

    vars = [cfilt1, cbias1, cfilt2, cbias2, FCweights, FCbias]

    return vars

###


def run_CNN(CIFAR_img,label,variables):
    true = np.zeros(10)
    true[label] = 1
    (cfilt1, cbias1, cfilt2, cbias2, FCweights, FCbias) = variables

    class var:
        val = []
        grad = []

    input = var()
    input.val = CIFAR_img
    input.grad = np.zeros_like(input.val)

    pad = 2
    stride = 1
    d = 2  ## receptive field for non overlapping maxpool

    pimg = f.pad_image(input, pad)
    # print('pimg mean,', np.mean(pimg.val))
    # print('pimg std', np.std(pimg.val))
    conv1 = f.convolve(pimg, cfilt1,cbias1, stride)
    # print('conv1 mean,', np.mean(conv1.val))
    # print('conv1 std', np.std(conv1.val))
    relu1 = f.relu(conv1)
    # print('relu1 mean,', np.mean(relu1.val))
    # print('relu1 std', np.std(relu1.val))
    maxpooled1 = f.maxpool(relu1, d)
    # print('maxpooled1 mean,', np.mean(maxpooled1.val))
    # print('maxpooled1 std', np.std(maxpooled1.val))
    pimg2 = f.pad_image(maxpooled1,pad)
    # print('pimg2 mean,', np.mean(pimg2.val))
    # print('pimg2 std', np.std(pimg2.val))
    conv2 = f.convolve(pimg2,cfilt2,cbias2,stride)
    # print('conv2 mean,', np.mean(conv2.val))
    # print('conv2 std', np.std(conv2.val))
    relu2 = f.relu(conv2)
    # print('relu2 mean,', np.mean(relu2.val))
    # print('relu2 std', np.std(relu2.val))
    maxpooled2 = f.maxpool(relu2,d)
    # print('maxpooled2 mean,', np.mean(maxpooled2.val))
    # print('maxpooled2 std', np.std(maxpooled2.val))
    scores = f.fullyconnected(maxpooled2, FCweights, FCbias, (10))
    # print(scores.val)  ## scores

    ## THEN SOFTMAX LOSS
    scores.val -= np.max(scores.val) #normalize scores by making largest zero,rest negative
    # print(scores.val)
    softmax = var()
    softmax.val = np.exp(scores.val) / np.sum(np.exp(scores.val))
    softmax.grad = np.zeros_like(softmax.val)

    loss = var()
    loss.grad = 1
    loss.val = f.CLLoss(softmax.val, true)

    # print(softmax.val)
    # print(loss.val)
    # print('yey')

    ## THEN BACKPROP (JUST THE CONVOLUTION HERE)

    scores = b.gscores(scores, softmax, true)

    FCweights = b.gFCweight(FCweights, maxpooled2, scores)
    FCbias = b.gFCbias(FCbias, scores)
    maxpooled2 = b.gFCinput(maxpooled2, FCweights, scores)

    relu2 = b.gmaxpool(relu2,d,maxpooled2)

    conv2 = b.grelu(conv2,relu2)
    cfilt2 = b.gconvfilter(cfilt2,pimg2,conv2,stride)
    cbias2 = b.gconvbias(cbias2,conv2)

    pimg2 = b.gconvinput(pimg2,cfilt2,conv2,stride)
    maxpooled1 = b.gpad(maxpooled1,pimg2,pad)

    relu1 = b.gmaxpool(relu1, d, maxpooled1)
    conv1 = b.grelu(conv1, relu1)

    cbias1 = b.gconvbias(cbias1, conv1)
    cfilt1 = b.gconvfilter(cfilt1, pimg, conv1, stride)
    pimg = b.gconvinput(pimg, cfilt1, conv1, stride)

    vars = [cfilt1, cbias1, cfilt2, cbias2, FCweights, FCbias]

    correct = (np.argmax(scores) == label)

    return [vars,correct,loss]

def reinitgrads(vars):
    for var in vars:
        var.grad = np.zeros_like(var.val)
    return vars