import tensorflow as tf
import numpy as np
import os
import argparse
import time  # 导入time模块
from data_utils import *
from model import *

def evaluate_trajectory_prediction(model_dir, dataset, obs_len, pred_len):
    set_random_seed(42)
    tf.compat.v1.reset_default_graph()
    checkpoint = os.path.join(model_dir, "trained_model.ckpt")
    loaded_graph = tf.Graph()

    with loaded_graph.as_default():
        with tf.compat.v1.Session(graph=loaded_graph) as sess:
            # 恢复已训练的模型
            loader = tf.compat.v1.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)

            # 获取图中的输入/输出张量
            inputs = loaded_graph.get_tensor_by_name('trajectory_prediction/inputs:0')
            predictions = loaded_graph.get_tensor_by_name('trajectory_prediction/outputs/BiasAdd:0')  
            lr = loaded_graph.get_tensor_by_name('trajectory_prediction/learning_rate:0')

            # 加载验证数据
            encoding_model_dir = os.path.join('./Models/Trajectory_Encoding_models', dataset + '_PTE')
            _, _, val_ages, val_targets = encoder_PedTraj(encoding_model_dir, dataset, obs_len, pred_len)

            # 记录推理开始时间
            start_time = time.time()

            # 计算预测值
            pred_outputs = sess.run(predictions, {inputs: val_ages, lr: 0.0})  # Shape: [20, batch_size, pred_len, 2]

            # 记录推理结束时间
            end_time = time.time()

            # 计算推理时间
            inference_time = end_time - start_time
            print(f"Inference Time: {inference_time:.3f} seconds")

            # 计算 Best-of-20 ADE 和 FDE
            best_distances = np.min(np.linalg.norm(pred_outputs - val_targets[None, :, :, :], axis=3), axis=0)
            ade = np.mean(np.mean(best_distances, axis=1))
            fde = np.mean(best_distances[:, -1])

            print('Evaluation Results on ' + dataset + ' :')
            print('ADE (Average Displacement Error): {:.3f}'.format(ade))
            print('FDE (Final Displacement Error): {:.3f}'.format(fde))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., HOTEL)')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    obs_len = 8  # Example observation length, adjust as needed
    pred_len = 12  # Example prediction length, adjust as needed
    prediction_model_name = dataset_name + '_TPN'
    prediction_model_dir = os.path.join('./Models/Trajectory_Prediction_models', prediction_model_name)

    # Evaluate the trajectory prediction model
    evaluate_trajectory_prediction(prediction_model_dir, dataset_name, obs_len, pred_len)