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


def stable_softmax(x_):
    exps = np.exp(x_ - np.max(x_))
    sum_ = np.sum(exps)
    if sum_ > 0:
        return exps / sum_
    else:
        return exps


def delta_cross_entropy(x_, y):
    m = y.shape[0]
    grad = stable_softmax(x_)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad


def cross_entropy(X, y):
    m = y.shape[0]
    p = stable_softmax(X)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


M = 1
D = 8  # I_
N = 4  # O_
print("Testing SoftMax forward")
np.random.seed(1216)
x = np.random.normal(size=(M, D))
w = np.random.normal(size=(D, N))
b = np.random.normal(size=(1, N))
o = x @ w + b
print("size_type I={}, O={};".format(D, N))
dump("x", x)
dump("w", w)
dump("b", b)

dump("o", stable_softmax(o))
y = np.array([np.random.choice(a=N, size=1) for _ in range(M)])
dump("y", y)
grad = delta_cross_entropy(o, y)
dump("g", grad)
loss = cross_entropy(o, y)
print("T loss_ = {};".format(loss))


