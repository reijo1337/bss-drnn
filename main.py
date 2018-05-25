import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl, QDir
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import design
from Utils import Utils
from config import RunConfig
from dataloader import DataLoader
from model import Model

import tensorflow as tf


def setup_path():
    if not os.path.exists(RunConfig.RESULT_PATH):
        os.makedirs(RunConfig.RESULT_PATH)


class SeparatorApp(QtWidgets.QMainWindow, design.Ui_Dialog):
    def __init__(self):
        self.inputName = ''
        super().__init__()
        self.setupUi(self)

        # self.m_voiceBox.setEnabled(False)
        # self.m_musicBox.setEnabled(False)
        # self.m_startSeparation.setEnabled(False)

        self.m_chooseFile.clicked.connect(self.loadFile)
        self.m_startSeparation.clicked.connect(self.separate)

        # Source
        self.m_sourceSlider.setRange(0, 0)
        self.m_sourceSlider.sliderMoved.connect(self.setPosition)
        self.sourceMediaPlayer = QMediaPlayer(self)
        self.sourceMediaPlayer.setVolume(100)
        self.sourceMediaPlayer.positionChanged.connect(self.positionChanged)
        self.sourceMediaPlayer.durationChanged.connect(self.durationChanged)
        self.m_playSource.clicked.connect(self.playSource)
        self.m_pauseSource.clicked.connect(self.pauseSource)
        self.m_stopSource.clicked.connect(self.stopSource)

        # Voice
        self.m_voiceSlider.setRange(0, 0)
        self.m_voiceSlider.sliderMoved.connect(self.setVoicePosition)
        self.voiceMediaPlayer = QMediaPlayer(self)
        self.voiceMediaPlayer.setVolume(100)
        self.voiceMediaPlayer.positionChanged.connect(self.voicePositionChanged)
        self.voiceMediaPlayer.durationChanged.connect(self.voiceDurationChanged)
        self.m_playVoice.clicked.connect(self.playVoice)
        self.m_pauseVoice.clicked.connect(self.pauseVoice)
        self.m_stopVoice.clicked.connect(self.stopVoice)

    def loadFile(self):
        self.inputName, _ = QFileDialog.getOpenFileName(self, 'Open file',
                                                        QDir.homePath(), "WAVE files (*.wav)")
        self.m_startSeparation.setEnabled(True)
        self.sourceMediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.inputName)))

    def separate(self):
        setup_path()
        model = Model()
        with tf.Session(config=RunConfig.session_conf) as sess:
            # Initialized, Load state
            sess.run(tf.global_variables_initializer())
            model.load(sess, RunConfig.CKPT_PATH)
            data = DataLoader(RunConfig.DATA_PATH)
            mixedWav, wavfiles = data.loadOne(self.inputName)

            mixSpectogram = Utils.toSpectogram(mixedWav)
            mixMagnitude = Utils.toMagnitude(mixSpectogram)
            mixBatch = Utils.spectogramToBatch(mixMagnitude)
            mixPhase = Utils.spectogramToPhase(mixSpectogram)

            (predMusicMagnitude, predVoiceMagnitude) = sess.run(model(), feed_dict={model.input: mixBatch})

            seq_len = mixPhase.shape[-1]
            predMusicMagnitude = Utils.batchToSpectogram(predMusicMagnitude, 9)[:, :, :seq_len]
            predVoiceMagnitude = Utils.batchToSpectogram(predVoiceMagnitude, 9)[:, :, :seq_len]

            # Time-frequency masking
            maskMusic = Utils.softTimeFreqMask(predMusicMagnitude, predVoiceMagnitude)
            # mask_src1 = hard_time_freq_mask(pred_src1_mag, pred_src2_mag)
            maskVoice = 1. - maskMusic
            predMusicMagnitude = mixMagnitude * maskMusic
            predVoiceMagnitude = mixMagnitude * maskVoice

            predMusicWav = Utils.toWav(predMusicMagnitude, mixPhase)
            predVoiceWav = Utils.toWav(predVoiceMagnitude, mixPhase)

            wavfiles = [self.inputName]
            for i in range(len(wavfiles)):
                name = wavfiles[i].replace('/', '-').replace('.wav', '')
                Utils.writeWav(mixedWav[i], '{}/{}-{}'.format(RunConfig.RESULT_PATH, name, 'original'))
                Utils.writeWav(predMusicWav[i], '{}/{}-{}'.format(RunConfig.RESULT_PATH, name, 'music'))
                Utils.writeWav(predVoiceWav[i], '{}/{}-{}'.format(RunConfig.RESULT_PATH, name, 'voice'))
                self.sourceMediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile('{}/{}-{}'.format(RunConfig.RESULT_PATH, name, 'music'))))

            self.m_voiceBox.setEnabled(True)
            self.m_musicBox.setEnabled(True)

    def playSource(self):
        self.sourceMediaPlayer.play()

    def pauseSource(self):
        self.sourceMediaPlayer.pause()

    def stopSource(self):
        self.sourceMediaPlayer.stop()

    def setPosition(self, position):
        self.sourceMediaPlayer.setPosition(position)

    def positionChanged(self, position):
        self.m_sourceSlider.setValue(position)

    def durationChanged(self, duration):
        self.m_sourceSlider.setRange(0, duration)

    def playVoice(self):
        self.voiceMediaPlayer.play()

    def pauseVoice(self):
        self.voiceMediaPlayer.pause()

    def stopVoice(self):
        self.voiceMediaPlayer.stop()

    def setVoicePosition(self, position):
        self.voiceMediaPlayer.setPosition(position)

    def voicePositionChanged(self, position):
        self.m_voiceSlider.setValue(position)

    def voiceDurationChanged(self, duration):
        self.m_voiceSlider.setRange(0, duration)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SeparatorApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
