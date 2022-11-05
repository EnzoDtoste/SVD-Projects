# %%
import numpy as np

# %%
#Randomized SVD
def rSVD(X, r, q, p):
    # p - oversampling factor
    # Random Matrix (X_columns x (r + p))
    ny = X.shape[1]
    P = np.random.randn(ny, r + p)

    Z = X @ P

    # Power Iterations    
    for k in range(q):
        Z = X @ (X.T @ Z)
    
    # QR Descomposition
    # Q Orthogonal Matrix
    # R Upper Triangular Matrix
    Q, R = np.linalg.qr(Z, mode='reduced')
    
    Y = Q.T @ X
    
    # Much faster because Y shape is (r + p) x X_columns  
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    
    U = Q @ UY

    # S, VT are the same of the Y-SVD
    return U, S, VT

# %%
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

# %%
# |X1 - X2| and then sum all the components
def simil(x1, x2):

    sum = 0

    for i in range(len(x1)):
        sum += abs(x1[i] - x2[i])

    return sum

# %%
#Precalculated matrixes of the Netflix Database

#A has a size of 8GB if your RAM is not enough plis select the other choice
A = np.load('value.enr.npy')
#A = np.load('value.enr.npy', mmap_mode='r')

# Matrixes created using Randomized SVD
U = np.load('U2.enr.npy')
S = np.load('S2.enr.npy')
VT = np.load('VT2.enr.npy')
V = VT.T

# Projection of all vectors from A in V  "A @ V"
L = np.load('L.enr.npy')

# New User Vector with the data of movies he likes/ doesn´t like
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

# %%
# Projection of X in V
X_L = X @ V

# Array that will contain the users who like the same kind of movies as the new user
simils = []

# Sorted components of X_L
items = [X_L[0]]
# Reference to the original index of the component
indexes = [0]

# Add the components of X_L in items keeping a descending order
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


# He likes that movie that much as me or a little bit (0.01) less than me?
def it_likes(l, index):
    if l[indexes[index]] >= items[index] or items[index] - l[indexes[index]] <= 0.01:
        return True

    return False

# Add in simils the users that like the two main kind of movie preferred by the new user
for i in range(len(L)):
    if it_likes(L[i], 0) and it_likes(L[i], 1):
        simils.append(i)


# I keep only the users who gave the same ratings for the movies that the new user watch or if they don´t watch it 

indexes = simils
passs = []

for i in range(len(X)):
    if X[i] != 0:
        for j in indexes:
            if A[j][i] == 0 or A[j][i] == X[i]:
                passs.append(j)

        indexes = passs 
        passs = []

# Project X in U
Sd = np.diag(S)
X_proj = X @ V @ np.linalg.inv(Sd)

# Add similarity of each vector of U with X_proj (closer to 0 better), in the AVL
avl = Avl(simil(U[indexes[0]], X_proj), indexes[0])

for fila in range(1, len(indexes), 1):
    avl.add(simil(U[indexes[fila]], X_proj), indexes[fila])

# List of tuples order by similarity, ascending
sort = avl.toArray()
# I keep only the better 50
sort = sort[:50]

# New vector X that will contain the predicted data
new_X = []

for i in range(len(X)):
    if X[i] != 0:
         new_X.append(X[i])
    else:

        # to predict the rating 
        # sum the user rating multiplied by his similarity(inverse because the more important the user is, higher will be his contribution to the sum)

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

# Print Results

for i in range(len(new_X)):
    if new_X[i] == 1:
        print(str(i + 1) + ' + 1')
    if new_X[i] == -1:
        print(str(i + 1) + ' - 1')


