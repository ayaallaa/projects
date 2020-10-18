# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:30:37 2020

@author: Aya Gamal
"""

import glob
import os
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from scipy.io.wavfile import write

def detect_leading_silence(sound, silence_threshold=-45.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size
    return trim_ms

def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    
    # Create the target shape    
    N_pad = N_tar - shape[0]
   
    print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    print('shape',shape)
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((data,np.zeros(shape)))
        else:
            return np.hstack((data,np.zeros(shape)))
    else:
        return data

for filepath in glob.iglob('E:\GP\Dataset_after_preprocessing\step3_dataset_silence_removal_padding\مياه/*.wav'):
    print(filepath)
    sample_rate, wav = wavfile.read(filepath) 
    print('sample_rate',sample_rate)
    #detect silence:
    sample_rate=16000
    L=sample_rate
    if len(wav) > L:
                sound = AudioSegment.from_file(filepath, format="wav")
                start_trim = detect_leading_silence(sound)
                end_trim = detect_leading_silence(sound.reverse())
                duration = len(sound)
                print("Duration of original audio=",duration)
                trimmed_sound = sound[start_trim:duration-end_trim]
                duration_trimmed_audio= (len(trimmed_sound))
                print("Duration_Trimmed_Audio per seconds=",duration_trimmed_audio)
                #save audio after silence removal
                trimmed_sound.export(filepath, format="wav")
                sample_rate, wav = wavfile.read(filepath) 
                print('length of wav file=',len(wav))
                #if audio is still long(>1s) after first trimming:
                if duration_trimmed_audio>1000:
                    print("still long")
                    #os.remove(filepath)
                    i = np.random.randint(0, len(wav) - L)
                    wav = wav[i:(i+L)]
                    print('new wav',len(wav))
                    wavfile.write(filepath,L,wav.astype(np.dtype('i2')))
                else:
                    print("short audio so,pad with zeros")
                    # create duration of silence audio segment
                    sound_padded=pad_audio(wav,sample_rate,1)
                    print(len(sound_padded))
                    wavfile.write(filepath,L,sound_padded.astype(np.dtype('i2')))
            
    # If shorter then randomly add silence
    elif len(wav) < L:
                    sound_padded=pad_audio(wav,sample_rate,1)
                    print(len(sound_padded))
                    wavfile.write(filepath,sample_rate,sound_padded.astype(np.dtype('i2'))) 
    else:
                    wavfile.write(filepath,L,wav.astype(np.dtype('i2')))    