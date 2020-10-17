from python_speech_features import mfcc
import scipy.io.wavfile as wav

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.signal.windows import hann
import os
import glob
from scipy.io import wavfile

n_mfcc = 13
n_mels = 40
n_fft = 512 
hop_length = 160
fmin = 0
fmax = None
sr = 16000
#sr, y = wav.read("Chair_247.wav") 

#mfcc_speech = mfcc(signal=y, samplerate=sr , winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512,
#  lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
#
#mfcc_data= np.swapaxes(mfcc_speech, 0 ,1)


#
#fig = plt.figure(frameon=False)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower')
#ax.set_axis_off()
#fig.add_axes(ax)
#ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower')
#fig.savefig('output_edit.png')
def read_wav_file(x):
    # Read wavfile using scipy wavfile.read
    sr, wav = wavfile.read(x) 
    print(len(wav))

    return sr, wav
for filepath in glob.iglob('tree2/*.wav'):
   
    print(filepath) 
    file=os.path.basename(filepath)



    sr,wav = read_wav_file(filepath)
    mfcc_speech = mfcc(signal=wav, samplerate=sr , winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512,
                       lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)

    mfcc_data= np.swapaxes(mfcc_speech, 0 ,1)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower')
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower')
    savefile = 'tree2mfcc/'+str(os.path.splitext(file)[0]) + '_' +'mfcc'+'.png'   # file might need to be replaced by a string
    plt.savefig(savefile)