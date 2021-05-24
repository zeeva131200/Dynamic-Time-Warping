import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

series1 = pd.Series([1, 4, 5, 10, 9, 3, 2, 6, 8, 4])
series2 = pd.Series([1,4,10,9,3,2,6,8,4,4])


##Fill DTW Matrix
def fill_dtw_cost_matrix(s1, s2):
    l_s_1, l_s_2 = len(s1), len(s2)
    cost_matrix = np.zeros((l_s_1 + 1, l_s_2 + 1))
    for i in range(l_s_1 + 1):
        for j in range(l_s_2 + 1):
            cost_matrix[i, j] = np.inf
    cost_matrix[0, 0] = 0

    for i in range(1, l_s_1 + 1):
        for j in range(1, l_s_2 + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            # take last min from the window
            prev_min = np.min([cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1]])
            cost_matrix[i, j] = cost + prev_min
            print("cost",cost)
            print("prev min",prev_min)
    return cost_matrix


##Call DTW function
dtw_cost_matrix = fill_dtw_cost_matrix(series1, series2)
print(dtw_cost_matrix)

from dtw import dtw

# Here, we use L2 norm as the element comparison distance
l2_norm = lambda x, y: (x - y) ** 2

dist, cost_matrix, acc_cost_matrix, path = dtw(series1, series2, dist=l2_norm)

print(dist)

plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()