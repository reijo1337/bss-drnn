from Utils import Diff, Utils
from config import TrainConfig
from dataloader import DataLoader
from model import Model
from train import summaries
import tensorflow as tf


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

    mix, music, voice = data.loadNext(TrainConfig.SECONDS)
    mixSpectogram = Utils.toSpectogram(mix)
    mixMagnitude = Utils.toMagnitude(mixSpectogram)

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
    writer.close()