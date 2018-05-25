import os

import tensorflow as tf


# Model
class ModelConfig:
    SR = 16000
    L_FRAME = 1024           # parameters for STFT
    L_HOP = L_FRAME // 4     # also this
    SEQ_LEN = 4


# Train
class TrainConfig:
    CASE = str(ModelConfig.SEQ_LEN) + 'dsd100'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/train'
    DATA_PATH = os.path.join('dataset', 'dsd100')
    LR = 0.01
    MAX_EPOCHS = 10000
    NUM_WAVFILE = 1
    SECONDS = 60
    RE_TRAIN = False
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
    )


class RunConfig:
    CASE = str(ModelConfig.SEQ_LEN) + 'dsd100'
    CKPT_PATH = 'checkpoints/' + CASE
    RESULT_PATH = 'results_run/' + CASE
    DATA_PATH = 'input'
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False
    )


class EvalConfig:
    CASE = str(ModelConfig.SEQ_LEN) + 'dsd100'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    DATA_PATH = 'dataset/dsd100'
    GRIFFIN_LIM = False
    GRIFFIN_LIM_ITER = 1000
    NUM_EVAL = 9
    SECONDS = 33
    RE_EVAL = True
    EVAL_METRIC = True
    WRITE_RESULT = True
    RESULT_PATH = 'result_run'
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False
    )