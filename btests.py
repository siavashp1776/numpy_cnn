import feedforward as f
import backprop as b
import numpy as np

class var:
    val = []
    grad = []

val = np.asarray([i for i in range(1,10)]).reshape(3,3)
val =  np.dstack((val,val,val))

v = np.asarray([i for i in range(1,5)]).reshape(2,2)

og = v.reshape(2,2,1)

v2 = np.dstack((v,v,v))
v2 = v2.reshape(2,2,3,1)

input = var()
input.val = val
input.grad = np.zeros_like(input.val)

cbiasz = var()
cbiasz.val = np.zeros(1)
cbiasz.grad = np.zeros_like(cbiasz.val)

cbias = var()
cbias.val = np.ones(1)
cbias.grad = np.zeros_like(cbias.val)

cfiltz = var()
cfiltz.val = np.zeros([2,2])
cfiltz.grad = np.zeros_like(cfiltz.val)

cfilt = var()
cfilt.val = v2
cfilt.grad = np.zeros_like(cfilt.val)

pad = 2
stride = 1
d = 2  ## receptive field for non overlapping maxpool

output = f.convolve(input,cfilt,cbias,stride)
output.grad = og
goutput = b.gconvbias(cbias,output)

# print(goutput)