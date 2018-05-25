import os
import tensorflow as tf

from Utils import Utils
from model import Model
from config import RunConfig, ModelConfig
from dataloader import DataLoader


def run():
    # Model
    model = Model()
    with tf.Session(config=RunConfig.session_conf) as sess:
        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load(sess, RunConfig.CKPT_PATH)
        data = DataLoader(RunConfig.DATA_PATH)
        mixedWav, wavfiles = data.loadInput()

        # getting spectogram and magnitude of mixed signal
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

        for i in range(len(wavfiles)):
            name = wavfiles[i].replace('/', '-').replace('.wav', '')
            Utils.writeWav(mixedWav[i], '{}/{}-{}'.format(RunConfig.RESULT_PATH, name, 'original'))
            Utils.writeWav(predMusicWav[i], '{}/{}-{}'.format(RunConfig.RESULT_PATH, name, 'music'))
            Utils.writeWav(predVoiceWav[i], '{}/{}-{}'.format(RunConfig.RESULT_PATH, name, 'voice'))


def setup_path():
    if not os.path.exists(RunConfig.RESULT_PATH):
        os.makedirs(RunConfig.RESULT_PATH)


if __name__ == '__main__':
    setup_path()