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

n = 32
ks = 16
d = 8
np.random.seed(1016)
data = np.random.normal(size=(n, d))
centroids = data[:ks, :].copy()

from scipy.cluster.vq import vq, _vq
codes = vq(data, centroids)[0]
centroids = _vq.update_cluster_means(data, codes, ks)[0]

dump("data", data)
dump("codes", codes)
dump("centroids", centroids)