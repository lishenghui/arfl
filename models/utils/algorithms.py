import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_lists(a, b):
    cosine = cosine_similarity(np.array(a).reshape(1, -1), np.array(b).reshape(1, -1))

    return cosine


def bi_partitioning(data):
    m = len(data)
    s = np.argsort(-data.flatten())
    C = [[e] for e in list(range(m))]
    for i in range(m * m):
        i1 = s[i] // m
        i2 = s[i] % m
        c_temp = []
        idx = []
        for c in C:
            if i1 in c or i2 in c:
                c_temp.extend(c)
                idx.append(C.index(c))

        for index in sorted(idx, reverse=True):
            del C[index]
        C.append(c_temp)
        if len(C) == 2:
            return C

if __name__ == "__main__":
    data = np.array([
        [1, -0.5, 0.9, -0.9],
        [-0.5, 1, -0.5, 0.9],
        [0.9, -0.5, 1, -0.5],
        [-0.9, 0.9, -0.5, 1]
    ])
    result = bi_partitioning(data)
    print(result)
