import IPython
import librosa
from dtw import dtw
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

tester = ['yt2.wav']
dataset = ['J&T-M.wav','memohon maaf-M.wav']

from pydub import AudioSegment
from pydub.utils import make_chunks

#to split audio file
myaudio = AudioSegment.from_file("yt2.wav" , "wav")
chunk_length_ms = 1000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

files=[]
for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    print ("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")
    files.append(chunk_name)

#the library librosa automatically loads into NumPy array after being sampled
#sampling frequency , sample rate
# y1, sr1 = librosa.load(files[2])
# y2, sr2 = librosa.load(dataset[0])

#print(y1,sr1)

#print(mfcc1)
# - use MFCC instead of time series

words=[]
def checkMin(arr, count):
    minimum = min(arr)
    if(minimum[1]=="memohon maaf-M.wav" and minimum[0]>=45681 and minimum[0]<=45682):
        print("min :", minimum)
        print("The spoken word is - : memohon maaf")
        words.append("memohon maaf")
    elif (minimum[1]=="J&T-M.wav"and minimum[0]>=41036 and minimum[0]<=41037):
        print("min :", minimum)
        print("The spoken word is - : J&T")
        words.append("J and T")
    else:
        print("word not in database")
    count+=1
    load(count)


#display cost matrix & distance & time series graph
def cost(y11,sr11,y12,sr12,T1,T2):
    #calculate mfcc then use dtw to compare the mfcc
    y1=y11
    y2=y12
    sr1 = sr11
    sr2 = sr12

    mfcc1 = librosa.feature.mfcc(y1, sr1)
    mfcc2 = librosa.feature.mfcc(y2, sr2)


  # display time series graph

    samplingFrequency1, signalData1 = wavfile.read(T1)
    samplingFrequency2, signalData2 = wavfile.read(T2)

    # Plot the signal - read from wav file

    plt.subplot(211)
    plt.title('Spectrogram of T1')
    plt.plot(signalData1)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(212)
    plt.title('Spectrogram of T2')
    plt.plot(signalData2)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

#    plt.show()
    # print(T1,",",T2)

    # plt.plot(x, label='x')
    # plt.plot(y, label='y')
    # plt.title('Our two temporal sequences')
    # plt.legend()
    # plt.show()


    #calc cost
    x = np.array([mfcc1]).reshape(-1, 1)
    y = np.array([mfcc2]).reshape(-1, 1)
    dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

    #display cost matrix
    plt.imshow(cost.T, origin='lower', cmap='gray', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
#    plt.show()
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[1] - 0.5))

    return dist


arr=[]
def load(count):
    if(count<=15):
        print(count)
        arr.clear()
        for i in range(len(dataset)):
            T1 = files[count]
            T2 = dataset[i]
            y1, sr1 = librosa.load(T1)
            y2, sr2 = librosa.load(T2)
            print("x:",T1,"y",T2,)
            arr.append([cost(y1,sr1,y2,sr2,T1,T2),T2])
            # print('Normalized distance between the two sounds:', cost(y1,sr1,y2,sr2,T1,T2))
            # print(arr)

        checkMin(arr,count)
    if(count>=16):
        print("\nwords identified from the audio file are : ",words)

count =2
load(count)