import numpy as np

np.random.seed(808)

D = 8

def dump(var, x_):
    print("vector<T> ", var+"_", " = {", ", ".join(list(map(str, np.reshape(x_, -1)))), "}; ")

a = np.random.normal(size=(D))
b = np.random.normal(size=(D))
dump("a", a)
dump("b", b)
print("float norm = ", np.sum((a-b)**2), ";")

x = np.random.normal(size=(D))
dump("x", x)
n = x / np.linalg.norm(x)
dump("n", n)