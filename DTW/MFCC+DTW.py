import IPython
import librosa
from dtw import dtw
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

tester = ['youtube.wav','memohon (US-slower-M).wav','maaf (US-faster-M).wav','memohon (US-slower-F).wav']
dataset = ['memohon (US-normal-F).wav','memohon (US-normal-M).wav','maaf (US-normal-M).wav']

#the library librosa automatically loads into NumPy array after being sampled
#sampling frequency , sample rate
y1, sr1 = librosa.load(tester[0])
y2, sr2 = librosa.load(dataset[0])

#print(y1,sr1)

#print(mfcc1)
# - use MFCC instead of time series



def checkMin(arr):
    minimum = min(arr)
    print("min :",minimum)
    if(minimum[1]=="memohon (US-normal-F).wav"or minimum[1]=="memohon (US-normal-M).wav"):
        print("The spoken word is - : memohon")
    else:
        print("The spoken word is - : maaf")


#display cost matrix & distance & time series graph
def cost():
    #calculate mfcc then use dtw to compare the mfcc
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    mfcc2 = librosa.feature.mfcc(y2, sr2)

#   display time series graph
    x = np.array([mfcc1]).reshape(-1, 1)
    y = np.array([mfcc2]).reshape(-1, 1)
    plt.plot(x, label='x')
    plt.plot(y, label='y')
    plt.title('Our two temporal sequences')
    plt.legend()
    plt.show()

#    samplingFrequency1, signalData1 = wavfile.read(tester[1])
#    plt.title('Spectrogram')
#    plt.plot(signalData1)
#    plt.xlabel('Time')
#    plt.ylabel('Amplitude')

    #calc cost
    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

    #display cost matrix
    plt.imshow(cost.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.show()
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[1] - 0.5))

    #print ('Normalized distance between the two sounds:', dist)
    return dist

count =1
arr=[]
max=2
for i in range(len(dataset)):
    x = tester[2]
    y = dataset[i]
    y1, sr1 = librosa.load(x)
    y2, sr2 = librosa.load(y)
    print("x:",x,"y",y,"count",count)
    count += 1
    arr.append([cost(),y])
    print('Normalized distance between the two sounds:', cost())
    print(arr)

    if(count==4):
        checkMin(arr)





