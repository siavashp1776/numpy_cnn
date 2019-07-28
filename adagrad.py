import numpy as np

class var:
    val = []
    grad = []

var1 = var()
var1.val = np.array([1e-3],dtype = 'float64')
var1.grad = np.array([1e-5], dtype = 'float64')

e_g2 = {}
e_x2 = {}

vars = [var1]

for var,i in zip(vars,list(range(np.size(vars)))):
    e_g2.update({i:np.zeros_like(var.val)})
    e_x2.update({i:np.zeros_like(var.val)})

p = np.float(0.95)
e = np.float(1e-6)

def adaDeltaUpdate(vars,p,epsilon,e_g2,e_x2):
    for var,i in zip(vars,list(range(np.size(vars)))):
        e_g2[i] = p*e_g2[i]+(1-p)*np.power(var.grad,2)
        rms_gt = np.sqrt(e_g2[i]+epsilon)
        rms_delta_x = np.sqrt(e_x2[i]+epsilon)
        delta_x = -rms_delta_x*var.grad/rms_gt
        e_x2[i] = p*e_x2[i]+(1-p)*np.power(delta_x,2)
        var.val+=delta_x
        print('delta x')
        print(delta_x)
    return [vars,e_g2,e_x2]
valarr = []
gradarr = []
e_g2arr = []
e_x2arr = []

for i in range(100):
    print('iteration:  ',i)
    [vars,e_g2,e_x2] = adaDeltaUpdate(vars,p,e,e_g2,e_x2)
    print('var info:')
    for i in vars:
        print(i.val)
        print(i.grad)
    print('eg2')
    for i in e_g2:
        print(e_g2[i])
    print('ex2')
    for i in e_x2:
        print(e_x2[i])
    print('')

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
x = [i for i in range(np.size(lossArr))]
axes.plot(x, lossArr)
fig.savefig('Loss.png')