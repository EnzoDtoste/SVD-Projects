from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


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

        elif self.value > value:
            if self.left:
                self.left.add(value, fila)
            else:
                self.left = Avl(value, fila)
            
        else:
            self.filas.append(fila)

    def toArray(self):
        if not self.left and not self.right:
            return [(self.value, self.filas)]

        elif not self.left:
            l = [(self.value, self.filas)]
            l.extend(self.right.toArray())
            return l

        elif not self.right:
            l = self.left.toArray()
            l.append((self.value, self.filas))
            return l

        else:
            l = self.left.toArray()
            l.append((self.value, self.filas))
            l.extend(self.right.toArray())
            return l

def rSVD(X, r, q, p):
    ny = X.shape[1]
    P = np.random.randn(ny, r + p)

    Z = X @ P
    print('mul')
    for k in range(q):
        Z = X @ (X.T @ Z)
    print('q')
    Q, R = np.linalg.qr(Z, mode='reduced')
    print('qr')
    Y = Q.T @ X
    print('proyected')
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    print('svd')
    U = Q @ UY
    print('mul')

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

def fulfill_vector(X, A, U, So, VT):
    
    V = VT.T

    S = np.diag(So)

    X_proj = X @ V @ np.linalg.inv(S)

    simils = Avl(simil(U[0], X_proj), 0)

    for fila in range(1, len(U), 1):
         simils.add(simil(U[fila], X_proj), fila)

    sort = simils.toArray()

    sort = sort[:10]

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


'''A = np.array([
    np.array([1,0,2]),
    np.array([2,2,0]),
    np.array([0,0,3]),
    np.array([0,2,1]),
    ])

X = np.array([1,2,0])

n = fulfill_vector(X, 2, 0, A)

print(n)'''

#dicc = {}

#first = open('combined_data_1.txt', "r", encoding="utf-8")

#movie = 0

#line = first.readline(-1)

#while True:

#  line = first.readline(-1)

#  if line == '':
#     break

#  if line.__contains__(':'):
#    movie += 1
#    print(str(movie / 4498 * 100) + '%')
#    continue

#  elements = line.split(',')
#  person = int(elements[0]) - 1
  
#  if not person in dicc:
#     dicc[person] = []
#     for i in range(4499):
#        dicc[person].append(0)
  
#  rating = int(elements[1])
     
#  if rating <= 2:
#     dicc[person][movie] = -1

#  elif rating >= 4:
#     dicc[person][movie] = 1


#print(str(movie))
#print(len(dicc.keys()))

#values = []

#for v in dicc.values():
#   values.append(v)

#np.save('value.enr', values)

#print(len(values))
#print(len(values[0]))

#A = np.load('value.enr.npy')
#print('load')
#U, S, VT = rSVD(A, 5, 5, 0)

#np.save('U2.enr', U)
#np.save('S2.enr', S)
#np.save('VT2.enr', VT)

#print(S)

##################################################

#A = np.load('value.enr.npy')
#print('load')
#X = []

#for i in range(4499):
#   X.append(0)


#X[12] = 1
#X[47] = 1
#X[190] = 1
#X[289] = 1
#X[1343] = 1
#X[1802] = -1
#X[2456] = -1
#X[2916] = 1

#U = np.load('U2.enr.npy')
#S = np.load('S2.enr.npy')
#VT = np.load('VT2.enr.npy')
#print('load')

#new_X = fulfill_vector(X, A, U, S, VT)

#for i in range(len(new_X)):
#    if new_X[i] == 1:
#        print(str(i + 1) + ' + 1')
#    if new_X[i] == -1:
#        print(str(i + 1) + ' - 1')

#######################################################

#A = np.load('value.enr.npy')
#VT = np.load('VT2.enr.npy')
#print('load')

#V = VT.T

#L = A @ V
#print('mul')
#np.save('L.enr', L)

#########################################################

A = np.load('value.enr.npy')
U = np.load('U2.enr.npy')
S = np.load('S2.enr.npy')
VT = np.load('VT2.enr.npy')
V = VT.T

L = np.load('L.enr.npy')

print('load')

X = [0] * 4499

X[7] = -1
X[12] = 1
X[47] = 1
X[190] = 1
X[289] = 1
X[1343] = 1
X[1802] = -1
X[2456] = -1
X[2916] = 1


X_L = X @ V

simils = []

indexes = [0]
items = [X_L[0]]

for i in range(1, len(X_L), 1):
    added = False
    for j in range(len(items)):
        if X_L[i] > items[j]:
            items.insert(j, X_L[i])
            indexes.insert(j, i)
            added = True
            break

    if not added:
        items.append(X_L[i])
        indexes.append(i)


def it_likes(l, index):
    if l[indexes[index]] >= items[index] or items[index] - l[indexes[index]] <= 0.01:
        return True

    return False

for i in range(len(L)):
    if it_likes(L[i], 0) and it_likes(L[i], 1):
        simils.append(i)

print(len(simils))

indexes = simils
passs = []

for i in range(len(X)):
    if X[i] != 0:
        for j in indexes:
            if A[j][i] == 0 or A[j][i] == X[i]:
                passs.append(j)

        indexes = passs 
        passs = []

print(len(indexes))

S = np.diag(S)
X_proj = X @ V @ np.linalg.inv(S)

avl = Avl(simil(U[indexes[0]], X_proj), indexes[0])

for fila in range(1, len(indexes), 1):
    avl.add(simil(U[indexes[fila]], X_proj), indexes[fila])

sort = avl.toArray()
sort = sort[:100]

new_X = []

for i in range(len(X)):
    if X[i] != 0:
         new_X.append(X[i])
    else:
         sum = 0

         for s in sort:
             for f in s[1]:
                 sum += A[f][i] * (1 / s[0])
               
         if abs(sum) <= 0.1:
            new_X.append(0)
         elif sum > 0:
            new_X.append(1)
         elif sum < 0:
            new_X.append(-1)


for i in range(len(new_X)):
    if new_X[i] == 1:
        print(str(i + 1) + ' + 1')
    if new_X[i] == -1:
        print(str(i + 1) + ' - 1')