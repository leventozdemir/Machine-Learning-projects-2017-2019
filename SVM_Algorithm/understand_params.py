import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#a small gamma implies a high variance and a generalized fit which might cause overfitting.
#To illustrate this, let's apply RBF kernel with different values to a dataset

X = np.c_[# negative class
          (0.3, -0.8),
          (-1.5, -1),
          (-1.3, -0.8),
          (-1.1, -1.3),
          (-1.2, -0.3),
          (-1.3, -0.5),
          (-0.6, 1.1),
          (-1.4, 2.2),
          (1, 1),
          # positive class
          (1.3, 0.8),
          (1.2, 0.5),
          (0.2, -2),
          (0.5, -2.4),
          (0.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [-1] * 8 + [1] * 8
gamma_option = [1, 2, 4]
plt.figure(1, figsize=(4*len(gamma_option), 4))
for i, gamma in enumerate(gamma_option, 1):
    svm = SVC(kernel='rbf', gamma=gamma)
    svm.fit(X, Y)
    plt.subplot(1, len(gamma_option), i)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10)
    plt.axes = ('tight')
    XX, YY = np.mgrid[-3:3:200j, -3:3:200j]
    Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    plt.title('gamma = %d' % gamma)

plt.show()