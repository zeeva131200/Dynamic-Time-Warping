from scipy.io.wavfile import read
import numpy as np
from dtw import dtw
samplerate1, data1 = read('memohon (US-normal-M).wav')
samplerate2, data2 = read('memohon (US-normal-F).wav')

#sampling frequency(sample rate) is no of data points per sec in a sound
print(samplerate1)
print(samplerate2)

print(data1)
print(data2)

duration1 = len(data1)/samplerate1
time1 = np.arange(0,duration1,1/samplerate1)
duration2 = len(data2)/samplerate2
time2 = np.arange(0,duration2,1/samplerate2)

import matplotlib.pyplot as plt
# plt.subplot(211)
# plt.plot(time1,data1)
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('memohon M Normal.wav')
#
# plt.subplot(212)
# plt.plot(time2,data2)
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.title('memohon F Normal.wav')
#plt.show()


x = np.array(data1).reshape(-1, 1)
y = np.array(data2).reshape(-1, 1)
plt.plot(x, label='x')
plt.plot(y, label='y')
plt.title('Our two temporal sequences')
# plt.legend()
# plt.show()

#Here, we use L2 norm as the element comparison distance
l2_norm = lambda x, y: (x - y) ** 2

dist, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=l2_norm)

print(dist)

plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()