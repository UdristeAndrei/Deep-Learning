from utils import Layer, LossFunction
from activation_fct import ReLu
import numpy as np


v = np.random.sample(size=10)
l1 = Layer(10, 5)
r1 = ReLu()
loss_fct = LossFunction()


o = l1.forward(v)
print(o)
o = r1.forward(o)
print(o)
l = loss_fct.forward(o, np.random.sample(len(o)))
print(l)

g_loss = loss_fct.backward()
print(g_loss)
g_r = r1.backward(g_loss)
print(g_r)
g_w = l1.backward(g_r)
print(g_w)
