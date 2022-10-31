from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import os


class Avl:

    def __init__(self, value, fila):
      self.value = value
      self.left = None
      self.right = None
      self.filas = [fila]

    def add(self, value, fila):
        if self.value < value:
            if self.right:
                self.right.add(value, fila)
            else:
                self.right = Avl(value, fila)

        elif self.value == value:
            self.filas.append(fila)

        else:
            if self.left:
                self.left.add(value, fila)
            else:
                self.left = Avl(value, fila)

    def toArray(self):
        if not self.left and not self.right:
            return self.filas

        elif not self.left:
            l = self.filas
            l.extend(self.right.toArray())
            return l

        elif not self.right:
            l = self.left.toArray()
            l.extend(self.filas)
            return l

        else:
            l = self.left.toArray()
            l.extend(self.filas)
            l.extend(self.right.toArray())
            return l

def rSVD(X, r, q, p):
    ny = X.shape[1]
    P = np.random.randn(ny, r + p)

    Z = X @ P

    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z, mode='reduced')

    Y = Q.T @ X

    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY

    return U, S, VT

def scale_image(path):
  i = imread(path)
  X = np.mean(i, -1);

  img = plt.imshow(X)
  img.set_cmap('gray')
  plt.axis('off')
  plt.show()

  U, S, VT = rSVD(X, 200, 2, 10)
  S = np.diag(S)

  j = 0

  for r in (100, 200):
      Xapproch = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
      plt.figure(j+1)
      j+= 1

      img = plt.imshow(Xapproch)
      img.set_cmap('gray')
      plt.axis('off')
      plt.title('r = ' + str(r))

      plt.show()
      
def simil(x1, x2):

    sum = 0

    for i in range(len(x1)):
        sum += abs(x1[i] - x2[i])

    return sum

def fulfill_vector(X, r, p, A):

    U, S, VT = rSVD(A, r, 2, p)

    V = VT.T

    S = np.diag(S)

    X_proj = X @ V @ np.linalg.inv(S)

    simils = Avl(simil(U[0], X_proj), 0)

    for fila in range(1, len(U), 1):
         simils.add(simil(U[fila], X_proj), fila)

    sort = simils.toArray()

    new_X = []

    for i in range(len(X)):
        if X[i] != 0:
            new_X.append(X[i])
        else:
            enter = False
            for f in sort:
                if A[f][i] != 0:
                    new_X.append(A[f][i])
                    enter = True
                    break
            
            if not enter:
                new_X.append(0)


    return new_X


A = np.array([
    np.array([1,0,2]),
    np.array([2,2,0]),
    np.array([0,0,3]),
    np.array([0,2,1]),
    ])

X = np.array([1,2,0])

n = fulfill_vector(X, 2, 0, A)

print(n)



