import numpy as np

M = 1
D = 2  # I_
N = 2  # O_

np.random.seed(808)


def dump(var, x_):
    print("vector<T> ", var+"_", " = {", ", ".join(list(map(str, np.reshape(x_, -1)))), "}; ")


x = np.random.normal(size=(M, D))
w = np.random.normal(size=(D, N))
b = np.random.normal(size=(1, N))
g = np.random.normal(size=(M, N))
dump("x", x)
dump("w", w)
dump("b", b)
dump("g", g)

o = x @ w + b
gx = g @ w.T
gw = x.T @ g
gb = np.sum(g, axis=0)
dump("o", o)
dump("gx", gx)

lr = 0.1
dump("gw", gw)
dump("gb", gb)

dump("update_w", w - lr*gw)
dump("update_b", b - lr*gb)
