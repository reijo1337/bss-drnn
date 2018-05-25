import os
import shutil

from Utils import Utils
from config import EvalConfig, ModelConfig
from dataloader import DataLoader
from mir_eval.separation import bss_eval_sources
from model import Model

import tensorflow as tf
import numpy as np


def eval():
    # Model
    model = Model()
    gStep = tf.Variable(0, dtype=tf.int32, trainable=False, name='gStep')

    with tf.Session(config=EvalConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load(sess, EvalConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(EvalConfig.GRAPH_PATH, sess.graph)

        data = DataLoader(EvalConfig.DATA_PATH)
        mix, music, voice, wavfiles = data.loadForEval(EvalConfig.SECONDS, EvalConfig.NUM_EVAL)

        mixSpectogram = Utils.toSpectogram(mix)
        mixMagnitude = Utils.toMagnitude(mixSpectogram)
        mixBatch = Utils.spectogramToBatch(mixMagnitude)
        mixPhase = Utils.spectogramToPhase(mixSpectogram)

        (musicMagnitudePred, voiceMagnitudePred) = sess.run(model(), feed_dict={model.input: mixBatch})

        seqLen = mixPhase.shape[-1]
        musicMagnitudePred = Utils.batchToSpectogram(musicMagnitudePred, EvalConfig.NUM_EVAL)[:, :, :seqLen]
        voiceMagnitudePred = Utils.batchToSpectogram(voiceMagnitudePred, EvalConfig.NUM_EVAL)[:, :, :seqLen]

        # Time-frequency masking
        maskMusic = Utils.softTimeFreqMask(musicMagnitudePred, voiceMagnitudePred)
        maskVoice = 1. - maskMusic
        musicMagnitudePred = mixMagnitude * maskMusic
        voiceMagnitudePred = mixMagnitude * maskVoice

        predMusicWav= Utils.toWav(musicMagnitudePred, mixPhase)
        predVoiceWav = Utils.toWav(voiceMagnitudePred, mixPhase)

        # Write the result
        tf.summary.audio('GT_mixed', mix, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)
        tf.summary.audio('Pred_music', predMusicWav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)
        tf.summary.audio('Pred_vocal', predVoiceWav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)

        if EvalConfig.EVAL_METRIC:
            # Compute BSS metrics
            gnsdr, gsir, gsar = bss_eval_global(mix, music, voice, predMusicWav, predVoiceWav)

            # Write the score of BSS metrics
            tf.summary.scalar('GNSDR_music', gnsdr[0])
            tf.summary.scalar('GSIR_music', gsir[0])
            tf.summary.scalar('GSAR_music', gsar[0])
            tf.summary.scalar('GNSDR_vocal', gnsdr[1])
            tf.summary.scalar('GSIR_vocal', gsir[1])
            tf.summary.scalar('GSAR_vocal', gsar[1])

        if EvalConfig.WRITE_RESULT:
            # Write the result
            for i in range(len(wavfiles)):
                name = wavfiles[i].replace('/', '-').replace('.wav', '')
                Utils.writeWav(mix[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'original'))
                Utils.writeWav(predMusicWav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'music'))
                Utils.writeWav(predVoiceWav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'voice'))

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=gStep.eval())

        writer.close()


def setup_path():
    if EvalConfig.RE_EVAL:
        if os.path.exists(EvalConfig.GRAPH_PATH):
            shutil.rmtree(EvalConfig.GRAPH_PATH)
        if os.path.exists(EvalConfig.RESULT_PATH):
            shutil.rmtree(EvalConfig.RESULT_PATH)

    if not os.path.exists(EvalConfig.RESULT_PATH):
        os.makedirs(EvalConfig.RESULT_PATH)


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:, :len_cropped]
    src2_wav = src2_wav[:, :len_cropped]
    mixed_wav = mixed_wav[:, :len_cropped]
    gnsdr = gsir = gsar = np.zeros(2)
    total_len = 0
    for i in range(EvalConfig.NUM_EVAL):
        sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav[i], src2_wav[i]]),
                                            np.array([pred_src1_wav[i], pred_src2_wav[i]]), False)
        sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav[i], src2_wav[i]]),
                                              np.array([mixed_wav[i], mixed_wav[i]]), False)
        nsdr = sdr - sdr_mixed
        gnsdr += len_cropped * nsdr
        gsir += len_cropped * sir
        gsar += len_cropped * sar
        total_len += len_cropped
    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    return gnsdr, gsir, gsar


if __name__ == '__main__':
    setup_path()
    eval()