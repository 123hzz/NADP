import tensorflow as tf
import numpy as np
import os
import csv
import random

# 关闭 eager 执行，使用 TF1 风格
tf.compat.v1.disable_eager_execution()

def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)            # 修改自 tf.set_random_seed
    random.seed(seed)

# Number of Epochs
epochs = 500
batch_size = 128
# RNN Size k = 256
rnn_size = 256
# Number of Layers, 2-layer LSTM
num_layers = 2
obs_len = 8
pred_len = 12
time_steps = obs_len
# For (x, y) coordinates
series_length = 2
learning_rate = 0.001
lr_decay = 0.95
momentum = 0.7
lambda_l2_reg = 0.01

tf.compat.v1.app.flags.DEFINE_string('dataset', 'ETH', "Dataset: ETH or HOTEL or UNIV or ZARA1 or ZARA2 or SDD")
tf.compat.v1.app.flags.DEFINE_string('gpu', '5', "GPU number")
FLAGS = tf.compat.v1.app.flags.FLAGS
config = tf.compat.v1.ConfigProto()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
config.gpu_options.allow_growth = True

# 全局变量
dataset = FLAGS.dataset
gpu = FLAGS.gpu

def get_inputs():
    input_data = tf.compat.v1.placeholder(tf.float32, [None, obs_len, series_length], name='inputs')  # 形状为 [None, 8, 2]
    targets = tf.compat.v1.placeholder(tf.float32, [None, pred_len, series_length], name='targets')   # 形状为 [None, 12, 2]
    lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
    target_sequence_length = tf.compat.v1.placeholder(tf.int32, (None,), name='target_sequence_length')
    source_sequence_length = tf.compat.v1.placeholder(tf.int32, (None,), name='source_sequence_length')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    return input_data, targets, lr, target_sequence_length, source_sequence_length, keep_prob

def get_data_ETH_UCY(dataset_name, obs_len, pred_len):
    set_random_seed(42)
    data = []
    filepath = os.path.join('/remote-home/hezhenzhen/PTE2-2080-tf2.4.0-py3.8/Datasets/ETH_UCY', dataset_name + '.csv')
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            frame = int(row[0])
            ped_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            data.append((frame, ped_id, x, y))
    
    # Organize data by pedestrian IDs
    ped_trajs = {}
    for frame, ped_id, x, y in data:
        if ped_id not in ped_trajs:
            ped_trajs[ped_id] = []
        ped_trajs[ped_id].append((frame, x, y))
    
    # Sort trajectories by frame and prepare sequences
    input_data = []
    targets = []
    for ped_id in ped_trajs:
        ped_trajs[ped_id].sort(key=lambda tup: tup[0])
        traj = ped_trajs[ped_id]
        traj_len = len(traj)
        if traj_len >= obs_len + pred_len:
            for idx in range(traj_len - obs_len - pred_len + 1):
                obs_seq = traj[idx:idx+obs_len]
                fut_seq = traj[idx+obs_len:idx+obs_len+pred_len]
                obs_positions = np.array([[x, y] for frame, x, y in obs_seq])
                fut_positions = np.array([[x, y] for frame, x, y in fut_seq])
                input_data.append(obs_positions)
                targets.append(fut_positions)

    # Convert to numpy arrays and shuffle
    input_data = np.array(input_data)
    targets = np.array(targets)
    indices = np.arange(input_data.shape[0])
    np.random.shuffle(indices)
    input_data = input_data[indices]
    targets = targets[indices]

    # Split into training and testing sets (80% training, 20% validation)
    split_idx = int(input_data.shape[0] * 0.8)
    train_input_data = input_data[:split_idx]
    train_targets = targets[:split_idx]
    val_input_data = input_data[split_idx:]
    val_targets = targets[split_idx:]

    return train_input_data, train_targets, val_input_data, val_targets

def pad_batch(batch_data, pad_int=0, pad_direction='post'):
    '''
    根据指定方向填充序列，适应轨迹预测模型需求。
    - batch_data: 需要填充的批次数据
    - pad_int: 填充值（默认为0）
    - pad_direction: 填充方向，'post'为后向填充，'pre'为前向填充
    '''
    max_len = max(len(sequence) for sequence in batch_data)
    padded_batch = []
    for sequence in batch_data:
        if pad_direction == 'post':
            padded_sequence = sequence + [[pad_int] * series_length] * (max_len - len(sequence))
        elif pad_direction == 'pre':
            padded_sequence = [[pad_int] * series_length] * (max_len - len(sequence)) + sequence
        else:
            raise ValueError("pad_direction should be either 'pre' or 'post'")
        padded_batch.append(padded_sequence)

    return np.array(padded_batch)

def get_batches(targets, sources, batch_size, source_pad_int=0, target_pad_int=0, pad_direction='post'):
    num_batches = len(sources) // batch_size
    for batch_i in range(num_batches):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        pad_sources_batch = np.array(pad_batch(sources_batch, source_pad_int, pad_direction))
        pad_targets_batch = np.array(pad_batch(targets_batch, target_pad_int, pad_direction))

        targets_lengths = [len(target) for target in targets_batch]
        source_lengths = [len(source) for source in sources_batch]

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

    # 处理不完整的最后一批数据
    if len(sources) % batch_size != 0:
        sources_batch = sources[num_batches * batch_size:]
        targets_batch = targets[num_batches * batch_size:]

        # 填充最后一批数据至 batch_size 大小
        while len(sources_batch) < batch_size:
            sources_batch.append([[source_pad_int] * series_length] * len(sources_batch[0]))
            targets_batch.append([[target_pad_int] * series_length] * len(targets_batch[0]))

        pad_sources_batch = np.array(pad_batch(sources_batch, source_pad_int, pad_direction))
        pad_targets_batch = np.array(pad_batch(targets_batch, target_pad_int, pad_direction))

        targets_lengths = [len(target) for target in targets_batch]
        source_lengths = [len(source) for source in sources_batch]

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths