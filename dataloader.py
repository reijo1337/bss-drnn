import os
import random

import librosa
import numpy as np

from config import ModelConfig


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.inputWavs = []
        for (path, dirs, files) in os.walk(self.path):
            self.inputWavs.extend(['{}/{}'.format(path, f) for f in files if f.endswith(".wav")])
        self.index = 0

    def loadNext(self, sec):
        if self.index == len(self.inputWavs):
            self.index = 0

        input = self.inputWavs[self.index]

        inputFile = [sampleRange(padWav(librosa.load(f, sr=ModelConfig.SR, mono=False)[0],
                                                     ModelConfig.SR, sec), ModelConfig.SR, sec) for f in [input]]

        mixed = np.array([librosa.to_mono(f) for f in inputFile])
        inputFile = np.array(inputFile)
        music, voice = inputFile[:, 0], inputFile[:, 1]
        self.index += 1
        return mixed, music, voice

    def loadForEval(self, sec, size):
        vawfiles = random.sample(self.inputWavs, size)
        inputFile = [sampleRange(padWav(librosa.load(f, sr=ModelConfig.SR, mono=False)[0],
                                        ModelConfig.SR, sec), ModelConfig.SR, sec) for f in vawfiles]

        mixed = np.array([librosa.to_mono(f) for f in inputFile])
        inputFile = np.array(inputFile)
        music, voice = inputFile[:, 0], inputFile[:, 1]
        return mixed, music, voice, vawfiles


    def loadOne(self, filename):
        wavfiles = [filename]
        mixed = [librosa.load(f, sr=ModelConfig.SR, mono=True)[0] for f in wavfiles]
        return mixed, wavfiles


def padWav(wav, sr, duration):
    assert (wav.ndim <= 2)

    numSamples = sr * duration
    padLen = np.maximum(0, numSamples - wav.shape[-1])
    if wav.ndim == 1:
        padWidth = (0, padLen)
    else:
        padWidth = ((0, 0), (0, padLen))
    wav = np.pad(wav, pad_width=padWidth, mode='constant', constant_values=0)

    return wav


def sampleRange(wav, sr, duration):
    assert (wav.ndim <= 2)

    targetLen = sr * duration
    wavLen = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wavLen - targetLen)), 1)[0]
    end = start + targetLen
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav
