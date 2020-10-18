# -*- coding: utf-8 -*-
"""
Created on Fri May 15 06:05:37 2020

@author: Hager
"""

import os
# os.environ["MODEL_DIR"] = '../model'
# os.environ["DATA_DIR"] = '../data'
os.environ["DATA_DIR"] = ''

import nlpaug
nlpaug.__version__


import nlpaug.augmenter.audio as naa
from nlpaug.util.audio import AudioVisualizer

import librosa
import librosa.display as librosa_display
import matplotlib.pyplot as plt

file_path = os.environ.get("DATA_DIR") + '1.wav'
data, sr = librosa.load(file_path)

## Crop Augmenter
aug = naa.CropAug(sampling_rate=sr)
augmented_data = aug.augment(data)

librosa_display.waveplot(augmented_data, sr=sr, alpha=0.5)
librosa_display.waveplot(data, sr=sr, color='r', alpha=0.25)

plt.tight_layout()
plt.show()


## Loudness Augmenter
aug = naa.LoudnessAug(loudness_factor=(2, 5))
augmented_data = aug.augment(data)

librosa_display.waveplot(augmented_data, sr=sr, alpha=0.25)
librosa_display.waveplot(data, sr=sr, color='r', alpha=0.5)

plt.tight_layout()
plt.show()

## Mask Augmenter
aug = naa.MaskAug(sampling_rate=sr, mask_with_noise=False)
augmented_data = aug.augment(data)

librosa_display.waveplot(data, sr=sr, alpha=0.5)
librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

plt.tight_layout()
plt.show()

## Noise             ##### Error
aug = naa.NoiseAug(noise_factor=0.03)
augmented_data = aug.augment(data)

librosa_display.waveplot(data, sr=sr, alpha=0.5)
librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

plt.tight_layout()
plt.show()

## Pitch Augmenter  ###### Error 
aug = naa.PitchAug(sampling_rate=sr, pitch_factor=(2,3))
augmented_data = aug.augment(data)

librosa_display.waveplot(data, sr=sr, alpha=0.5)
librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

plt.tight_layout()
plt.show()

## Shift Augmenter
aug = naa.ShiftAug(sampling_rate=sr)
augmented_data = aug.augment(data)

librosa_display.waveplot(data, sr=sr, alpha=0.5)
librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

plt.tight_layout()
plt.show()

## Speed Augmenter
aug = naa.SpeedAug()
augmented_data = aug.augment(data)

librosa_display.waveplot(data, sr=sr, alpha=0.5)
librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

plt.tight_layout()
plt.show()

## VTLP Augmenter
aug = naa.VtlpAug(sampling_rate=sr)
augmented_data = aug.augment(data)

VisualWave.freq_power('VTLP Augmenter', data, sr, augmented_data) #### Error
