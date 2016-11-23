import random


def UniSample(K):
    return random.randint(0, K - 1)


def MultSample(ProbList):
    size = len(ProbList)
    for i in xrange(1, size):
        ProbList[i] += ProbList[i - 1]
    u = random.random()
    res = 0
    for k in xrange(size):
        if ProbList[k] >= u * ProbList[size - 1]:
            res = k
            break
    return res
