# import the pyplot and wavfile modules

import matplotlib.pyplot as plot

from scipy.io import wavfile
from dtw import dtw

# Read the wav file (mono)

samplingFrequency1, signalData1 = wavfile.read('chunk2.wav')
samplingFrequency2, signalData2 = wavfile.read('J&T-M.wav')

# Plot the signal read from wav file
x= str('chunk2.wav')
y=str('J&T-M.wav')

plot.subplot(211)
plot.title('Spectrogram of',x)
plot.plot(signalData1)
plot.xlabel('Time')
plot.ylabel('Amplitude')

plot.subplot(212)
plot.title('Spectrogram of ',y)
plot.plot(signalData2)
plot.xlabel('Time')
plot.ylabel('Amplitude')

plot.show()
# plot.subplot(212)
#
# plot.specgram(signalData, Fs=samplingFrequency)
#
# plot.xlabel('Time')
#
# plot.ylabel('Frequency')


