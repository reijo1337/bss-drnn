# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell

from config import ModelConfig
import numpy as np
import os


class Model:
    def __init__(self, hiddenSize=256, nRnnLayer=3):
        # Input and output shapes
        inputShape = (None, None, ModelConfig.L_FRAME // 2 + 1)
        outputMusicShape = (None, None, ModelConfig.L_FRAME // 2 + 1)
        outputVoiceShape = (None, None, ModelConfig.L_FRAME // 2 + 1)

        # TF graph input
        self.input = tf.placeholder(tf.float32, shape=inputShape, name='mix')
        # TF graph output
        self.music = tf.placeholder(tf.float32, shape=outputMusicShape, name='music')
        self.voice = tf.placeholder(tf.float32, shape=outputVoiceShape, name='voice')

        # Network
        self.hiddenSize = hiddenSize  # Size of hidden layers
        self.layerCount = nRnnLayer  # Count of hidden layers
        self.network = tf.make_template('network', self.netGen)
        self()

    def netGen(self):
        # Creating of N=layerCount RNN layers
        rnnLayer = MultiRNNCell([GRUCell(self.hiddenSize) for _ in range(self.layerCount)])
        # Creating of RNN network
        outputRnn, _ = tf.nn.dynamic_rnn(rnnLayer, self.input, dtype=tf.float32)
        inputSize = np.shape(self.input)[2]

        # Dense layer
        musicHat = tf.layers.dense(inputs=outputRnn, units=inputSize, activation=tf.nn.relu, name='musicHat')
        voiceHat = tf.layers.dense(inputs=outputRnn, units=inputSize, activation=tf.nn.relu, name='voiceHat')

        # Time-frequency masking layer
        retMusic = musicHat / (musicHat + voiceHat + np.finfo(float).eps) * self.input
        retVoice = voiceHat / (musicHat + voiceHat + np.finfo(float).eps) * self.input

        return retMusic, retVoice

    def __call__(self):
        return self.network()

    def loss(self):
        musicPredicted, voicePredicted = self()
        return tf.reduce_mean(tf.square(self.music - musicPredicted) + tf.square(self.voice - voicePredicted), name='loss')

    @staticmethod
    def load(session, path):
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(path+'/checkpoint'))
        if checkpoint and checkpoint.model_checkpoint_path:
            tf.train.Saver().restore(session, checkpoint.model_checkpoint_path)

    @staticmethod
    def save(session, path, step):
        tf.train.Saver().save(session, path+'/checkpoint', global_step=step)