# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:39:34 2020

@author: Aya Gamal
"""

from scipy.io import wavfile
from pydub import AudioSegment
from scipy.io.wavfile import write
import glob
import os
import numpy as np
import samplerate 

for filepath in glob.iglob('E:\GP\Dataset_after_preprocessing\step2_dataset_resampling\data_with_samplerate16000\مياه/*.wav'):
    #print(filepath)
    # sample_rate, wav = wavfile.read(filepath) 
    # print('sample_rate',sample_rate)
    # if sample_rate==44100:
    #     print('A')
    #     sample_rate=48000
    #     resampled_wav = samplerate.resample(wav, 48000 / 44100, 'sinc_best')
    #     print(len(resampled_wav))
    #     wavfile.write(filepath,48000,resampled_wav.astype(np.dtype('i2')))
    # # elif sample_rate==44100:
    # #     print('B')
    # #     sample_rate=16000
    # #     resampled_wav = samplerate.resample(wav, 16000 / 44100, 'sinc_best')
    # #     print(len(resampled_wav))
    # #     wavfile.write(filepath,sample_rate,resampled_wav.astype(np.dtype('i2')))
    # else :
    #     print('B')



#for 16000 sampling rate 
        sample_rate, wav = wavfile.read(filepath) 
        print('sample_rate',sample_rate)
        if sample_rate==48000:
            print('A')
            sample_rate=16000
            resampled_wav = samplerate.resample(wav, 16000 / 48000, 'sinc_best')
            print(len(resampled_wav))
            wavfile.write(filepath,16000,resampled_wav.astype(np.dtype('i2')))
        elif sample_rate==44100:
            print('B')
            sample_rate=16000
            resampled_wav = samplerate.resample(wav, 16000 / 44100, 'sinc_best')
            print(len(resampled_wav))
            wavfile.write(filepath,sample_rate,resampled_wav.astype(np.dtype('i2')))
        else :
            print('C')
