import numpy as np
from scipy.cluster.vq import vq, _vq

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


def kmeans(x, iter_):
    assert iter_ > 0
    centers = x[:ks, :].copy()
    assigned = None
    for i in range(iter_):
        assigned = vq(x, centers)[0]
        centers = _vq.update_cluster_means(x, assigned, ks)[0]
    cx = centers[assigned]
    return assigned, centers, cx


codes, centroids, _ = kmeans(data, 1)


def rq(x, depth, iter):
    residual = x.copy()
    codes = []
    codewords = []
    for _ in range(depth):
        code, codeword, compressed = kmeans(residual, iter)
        assert residual.shape == compressed.shape
        residual -= compressed
        codes.append(code)
        codewords.append(codeword)

    return np.stack(codes, axis=0), np.stack(codewords, axis=0), x - residual


depth = 8
r_codes, r_centroids, r_compressed = rq(data, depth, 1)

dump("data", data)
dump("codes", codes)
dump("centroids", centroids)

dump("r_codes", r_codes)
dump("r_centroids", r_centroids)
