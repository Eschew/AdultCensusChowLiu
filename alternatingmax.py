import numpy as np
import math

def KL(p, q):
  assert p.shape[0] == q.shape[0]
  lg = lambda x: np.log(x, 2)

  kl = 0.
  for i in range(p.shape[0]):
    if p[i] == 0:
      continue
    assert q[i] != 0, "infinite KL"
    kl += p[i]*math.log(p[i]/q[i], 2)

  return kl

def KLMatrix(mat, q):
  l = np.zeros((mat.shape[0]))

  for i in range(mat.shape[0]):
    l[i] = KL(mat[i, :], q)

  return l



def findCapacity(mat, eps=0.001):
  r = np.ones(mat.shape[0])/mat.shape[0]
  qxy = np.zeros(mat.shape)
  prevC = -1

  while 1:
    for y in range(mat.shape[1]):
      total = 0
      qxy[y, :] = np.multiply(r, mat[:,y])
      qxy[y, :] = qxy[y, :]/np.sum(qxy[y, :])

    total = 0
    for x in range(r.shape[0]):
      t1 = 1
      for y in range(mat.shape[0]):
        t1 = (qxy[y, x])**mat[x,y]
      total += t1
      r[x] = t1

    r = r/total

    C = 0

    for x in range(mat.shape[0]):
      for y in range(mat.shape[1]):
        if qxy[y,x] == 0.:
          continue
        C += r[x]*mat[x,y]*math.log(qxy[y,x]/r[x],2)

    if np.abs(C-prevC) < eps:
      return C
    prevC = C


mat = np.array([[.3, .4, .3],[.9, 0, .1],[.1, .4, .5]])
# mat = np.array([[1., 0.], [0., 1.]])

print findCapacity(mat)


