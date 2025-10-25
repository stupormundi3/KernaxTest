import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import cm

# Je dois encore convertir en jax
# Suppose there is a circle with center at (0, 0) and radius 1.2.
# All the points within the circle are labeled 1.
# All the points outside the circle are labeled 0.
nSamples = 10
spanLen = 2
X = np.zeros((nSamples, 2))
y = np.zeros((nSamples, ))

for i in range(nSamples):
  a, b = [np.random.uniform(-spanLen, spanLen) for _ in ['x', 'y']]
  X[i][0], X[i][1] = a, b
  y[i] = 1 if a*a + b*b < 1.2*1.2 else 0

# Custom kernel,
def my_kernel(A, B):
  gram = np.zeros((A.shape[0], B.shape[0]))
  for i in range(A.shape[0]):
    for j in range(B.shape[0]):
      assert A.shape[1] == B.shape[1]
      L2A, L2B = 0.0, 0.0
      for k in range(A.shape[1]):
        gram[i, j] += A[i, k] * B[j, k]
        L2A += A[i, k] * A[i, k]
        L2B += B[j, k] * B[j, k]
      gram[i, j] += L2A * L2B 
  return gram

# SVM train.
clf = svm.SVC(kernel = my_kernel)
clf.fit(X, y)
coef = clf.dual_coef_[0]
sup = clf.support_
b = clf.intercept_
x_min, x_max = -spanLen, spanLen
y_min, y_max = -spanLen, spanLen
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the 2D layout.
fig = plt.figure(figsize = (6, 14))
plt1 = plt.subplot(121)
plt1.set_xlim([-spanLen, spanLen])
plt1.set_ylim([-spanLen, spanLen])
plt1.set_xticks([-1, 0, 1])
plt1.set_yticks([-1, 0, 1])
plt1.pcolormesh(xx, yy, Z, cmap=cm.Paired)
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
  this_Xx = [X[i][0] for i in range(len(X)) if y[i] == this_y]
  this_Xy = [X[i][1] for i in range(len(X)) if y[i] == this_y]
  plt1.scatter(this_Xx, this_Xy,  c=color.reshape(1,-1), alpha=0.5)

# Process the training data into 3D by applying the kernel mapping:
# phi(x, y) = (x, y, x*x + y*y).
X3d = np.ndarray((X.shape[0], 3))
for i in range(X.shape[0]):
    a, b = X[i][0], X[i][1]
    X3d[i, 0], X3d[i, 1], X3d[i, 2] = [a, b, a*a + b*b]

# Plot the 3D layout after applying the kernel mapping.
from mpl_toolkits.mplot3d import Axes3D
plt2 = plt.subplot(122, projection="3d")
plt2.set_xlim([-spanLen, spanLen])
plt2.set_ylim([-spanLen, spanLen])
plt2.set_xticks([-1, 0, 1])
plt2.set_yticks([-1, 0, 1])
plt2.set_zticks([0, 2, 4])
for this_y, color in zip(y_unique, colors):
  this_Xx = [X3d[i, 0] for i in range(len(X3d)) if y[i] == this_y]
  this_Xy = [X3d[i, 1] for i in range(len(X3d)) if y[i] == this_y]
  this_Xz = [X3d[i, 2] for i in range(len(X3d)) if y[i] == this_y]
  plt2.scatter(this_Xx, this_Xy, this_Xz, c=color.reshape(1,-1), alpha=0.5)

# Plot the 3D boundary.
def onBoundary(x, y, z, X3d, coef, sup, b):
  err = 0.0
  n = len(coef)
  for i in range(n):
    err += coef[i] * (x*X3d[sup[i], 0] + y*X3d[sup[i], 1] + z*X3d[sup[i], 2])
  err += b
  if abs(err) < .1:
    return True
  return False

Xr = np.arange(x_min, x_max, .02)
Yr = np.arange(y_min, y_max, .02)
Z = np.zeros(Z.shape)
for i in range(Xr.shape[0]):
  x = Xr[i]
  for j in range(Yr.shape[0]):
    y = Yr[j]
    for z in np.arange(0, 2, .02):
      if onBoundary(x, y, z, X3d, coef, sup, b):
        Z[i, j] = z
        break
plt2.plot_surface(xx, yy, Z, cmap='summer', alpha=0.2)

plt.savefig("kerneltrick.svg", format = "svg")