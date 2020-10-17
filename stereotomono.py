# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:55:54 2020

@author: Aya Gamal
"""
import glob
import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
for filepath in glob.iglob('E:\GP\Dataset_after_preprocessing\step3_dataset_silence_removal_padding\مياه/*.wav'):
    print(filepath)
    sample_rate, wav = wavfile.read(filepath) 
    sound = AudioSegment.from_wav(filepath)
    sound = sound.set_channels(1)
    sound.export(filepath, format="wav")