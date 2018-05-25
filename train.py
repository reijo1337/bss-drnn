from config import TrainConfig
from model import Model
from dataloader import DataLoader
from Utils import Utils, Diff
import os
import shutil

import tensorflow as tf


def setPath():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
        if os.path.exists(TrainConfig.GRAPH_PATH):
            shutil.rmtree(TrainConfig.GRAPH_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)


def train():
    model = Model()

    gStep = tf.Variable(0, dtype=tf.int32, trainable=False, name='gStep')
    loss = model.loss()
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=TrainConfig.LR).minimize(loss, global_step=gStep)

    summariesOp = summaries(model, loss)

    with tf.Session(config=TrainConfig.session_conf) as session:
        session.run(tf.global_variables_initializer())
        model.load(session, TrainConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, session.graph)

        data = DataLoader(TrainConfig.DATA_PATH)
        ls = Diff()

        for epoch in range(gStep.eval() + 10, TrainConfig.MAX_EPOCHS  + 10):
            for step in range(0, len(data.inputWavs)):
                # loading example from dataset
                mix, music, voice = data.loadNext(TrainConfig.SECONDS)

                # getting spectogram and magnitude of mixed signal
                mixSpectogram = Utils.toSpectogram(mix)
                mixMagnitude = Utils.toMagnitude(mixSpectogram)

                # getting spectograms and magnitudes of sources
                musicSpectogram = Utils.toSpectogram(music)
                musicMagnitude = Utils.toMagnitude(musicSpectogram)

                voiceSpectogram = Utils.toSpectogram(voice)
                voiceMagnitude = Utils.toMagnitude(voiceSpectogram)

                musicBatch = Utils.spectogramToBatch(musicMagnitude)
                voiceBatch = Utils.spectogramToBatch(voiceMagnitude)
                mixBatch = Utils.spectogramToBatch(mixMagnitude)

                l, _, summary = session.run([loss, optimizer, summariesOp],
                                            feed_dict={model.input: mixBatch,
                                                       model.music: musicBatch, model.voice: voiceBatch})
                ls.update(l)
                print('epoch-{} step-{}\td_loss={:2.2f}\tloss={}'.format(epoch, step, ls.diff * 100, ls.value))

                writer.add_summary(summary, global_step=epoch*len(data.inputWavs) + step)

            tf.train.Saver().save(session, TrainConfig.CKPT_PATH + '/checkpoint', global_step=len(data.inputWavs)*(epoch+1))

        writer.close()


# Export information about training process
def summaries(model, loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
        tf.summary.histogram('grad/' + v.name, tf.gradients(loss, v))
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('mixed', model.input)
    tf.summary.histogram('music', model.music)
    tf.summary.histogram('voice', model.voice)
    return tf.summary.merge_all()


if __name__ == "__main__":
    setPath()
    train()
