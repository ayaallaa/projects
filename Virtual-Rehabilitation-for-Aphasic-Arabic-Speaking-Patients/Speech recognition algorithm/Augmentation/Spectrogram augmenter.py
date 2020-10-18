# -*- coding: utf-8 -*-
"""
Created on Thr May 14 04:58:48 2020

@author: Hager
"""


from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util.audio.visualizer import AudioVisualizer
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf


path = '1.wav'

mel_spectrogram = AudioLoader.load_mel_spectrogram(path, n_mels=128)
AudioVisualizer.spectrogram('Original', mel_spectrogram)




## Frequency Masking
aug = nas.FrequencyMaskingAug(mask_factor=80)

augmented_mel_spectrogram = aug.substitute(mel_spectrogram)
AudioVisualizer.spectrogram('Frequency Masking', augmented_mel_spectrogram)


## Time Masking
aug = nas.TimeMaskingAug(mask_factor=80)

augmented_mel_spectrogram = aug.substitute(mel_spectrogram)
AudioVisualizer.spectrogram('Time Masking', augmented_mel_spectrogram)


## Combine Frequency Masking and Time Masking
flow = naf.Sequential([
    nas.FrequencyMaskingAug(mask_factor=50), 
    nas.TimeMaskingAug(mask_factor=20), 
    nas.TimeMaskingAug(mask_factor=30)])
augmented_mel_spectrogram = flow.augment(mel_spectrogram)
AudioVisualizer.spectrogram('Combine Frequency Masking and Time Masking', augmented_mel_spectrogram)