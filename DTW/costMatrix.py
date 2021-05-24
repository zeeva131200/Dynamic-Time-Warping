import matplotlib.pyplot as plt
import numpy as np
# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = np.array([1, 4, 5, 10, 9, 3, 2, 6, 8, 4]).reshape(-1, 1)
y = np.array([1, 7, 3, 4, 1, 10, 5, 4, 7, 4]).reshape(-1, 1)
#1, 7, 3, 4, 1, 10, 5, 4, 7, 4
#1,4,10,9,3,2,6,8,4,4
plt.plot(x, label='x')
plt.plot(y, label='y')
plt.title('Our two temporal sequences')
plt.legend()
plt.show()

from dtw import dtw

# Here, we use L2 norm as the element comparison distance
l2_norm = lambda x, y: (x - y) ** 2

#dist, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=l2_norm)
dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

print(dist)

# plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path[0], path[1], 'w')
# plt.show()

#lower distance indicates higher similiarity of the two sounds