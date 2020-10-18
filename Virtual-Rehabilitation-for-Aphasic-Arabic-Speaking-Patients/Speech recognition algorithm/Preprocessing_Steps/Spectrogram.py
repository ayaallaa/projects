# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:38:01 2019

@author: Aya Gamal
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import stft



def read_wav_file(x):
    # Read wavfile using scipy wavfile.read
    _, wav = wavfile.read(x) 
    print(len(wav))
    # Normalize
   # wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

def log_spectrogram(wav):
    freqs, times, spec = stft(wav, SAMPLE_RATE, nperseg = 400, noverlap = 240, nfft = 512, 
                              padded = False, boundary = None)
    # Log spectrogram
    amp = np.log(np.abs(spec)+1e-10)
    return freqs, times, amp

fig = plt.figure(figsize=(14, 8))   
for filepath in glob.iglob('tree2/*.wav'):
   
    print(filepath) 
    file=os.path.basename(filepath)
    SAMPLE_RATE = 16000
# fig = plt.figure(figsize=(14, 10))
# for i, fn in enumerate(fns):
#     wav = read_wav_file(fn)
 
#     ax = fig.add_subplot(3,1,i+1)
#     ax.set_title('Raw wave of ' + fn)
#     ax.set_ylabel('Amplitude')
#     ax.plot(np.linspace(0, SAMPLE_RATE/len(wav), SAMPLE_RATE), wav)
# fig.tight_layout()

    wav = read_wav_file(filepath)
    freqs, times, amp = log_spectrogram(wav)
    plt.imshow(amp, aspect='auto', origin='lower', 
                extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    savefile = 'tree2spectrograms/'+str(os.path.splitext(file)[0]) + '_' +'Spectrogram'+'.png'   # file might need to be replaced by a string
    plt.savefig(savefile)
fig.tight_layout()