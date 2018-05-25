from config import ModelConfig
import numpy as np
import librosa


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def toSpectogram(signal, lenFrame=ModelConfig.L_FRAME, lenHop=ModelConfig.L_HOP):
        return np.array([librosa.stft(s, n_fft=lenFrame, hop_length=lenHop) for s in signal])

    @staticmethod
    def toMagnitude(spectogram):
        return np.abs(spectogram)

    @staticmethod
    def spectogramToBatch(spec):
        numWavs, freq, numFrames = spec.shape

        # Padding
        padLen = 0
        if numFrames % ModelConfig.SEQ_LEN > 0:
            padLen = (ModelConfig.SEQ_LEN - (numFrames % ModelConfig.SEQ_LEN))
        padWidth = ((0, 0), (0, 0), (0, padLen))
        paddedSpec = np.pad(spec, pad_width=padWidth, mode='constant', constant_values=0)

        assert (paddedSpec.shape[-1] % ModelConfig.SEQ_LEN == 0)

        batch = np.reshape(paddedSpec.transpose(0, 2, 1), (-1, ModelConfig.SEQ_LEN, freq))
        return batch

    @staticmethod
    def spectogramToPhase(spec):
        return np.angle(spec)

    @staticmethod
    def batchToSpectogram(src, nWav):
        batchSize, seqLen, freq = src.shape
        src = np.reshape(src, (nWav, -1, freq))
        src = src.transpose(0, 2, 1)
        return src

    @staticmethod
    def softTimeFreqMask(targetSrc, remainingSrc):
        mask = np.abs(targetSrc) / (np.abs(targetSrc) + np.abs(remainingSrc) + np.finfo(float).eps)
        return mask

    @staticmethod
    def toWav(mag, phase, lenHop=ModelConfig.L_HOP):
        stftMaxrix = mag * np.exp(1.j * phase)
        return np.array([librosa.istft(s, hop_length=lenHop) for s in stftMaxrix])

    @staticmethod
    def writeWav(data, path, sr=ModelConfig.SR):
        # sf.write('{}.wav'.format(path), data, sr, format=format, subtype=subtype)
        librosa.output.write_wav('{}.wav'.format(path), data.astype(np.float32), sr)

    @staticmethod
    def softTimeFreqMask(target_src, remaining_src):
        mask = np.abs(target_src) / (np.abs(target_src) + np.abs(remaining_src) + np.finfo(float).eps)
        return mask


class Diff(object):
    def __init__(self, value=0.):
        self.value = value
        self.diff = 0.

    def update(self, value):
        if self.value:
            diff = (value / self.value - 1)
            self.diff = diff
        self.value = value