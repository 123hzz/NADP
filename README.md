![Python = 3.8](https://img.shields.io/badge/Python-=3.8-blue.svg)
![Tensorflow = 2.4.0](https://img.shields.io/badge/Tensorflow-=2.4.0-yellow.svg)
# Decoupled Pedestrian Trajectory Prediction Network with Near-Aware Attention
By Zhenzhen He,  Wenting Li, Xiaorong Gan, Ziyang Chen, Yabo Wu, Yongjun Zhang. In [KBS 2025](https://www.sciencedirect.com/journal/knowledge-based-systems) with open [codes](https://github.com/123hzz/NADP).


## Introduction
This is the official implementation of the decoupled pedestrian trajectory prediction network model presented by "Decoupled Pedestrian Trajectory Prediction Network with Near-Aware Attention".


Abstract: Pedestrian trajectory prediction estimates future pedestrian positions based on historical trajectories and is crucial for autonomous driving and robotics navigation. The existing methods, although improving prediction accuracy, often introduce redundant information or excessive updates to features, leading to increased model complexity and reduced inference speed. Simplified models, although faster, may overlook complex interactions, leading to reduced accuracy in dense scenes. To address this issue, we propose a decoupled pedestrian trajectory prediction network with near-aware attention (NADP). The network adopts a decoupled design (DD) strategy, dividing the network into two complementary sub-networks: the feature optimization network and the trajectory generation network. DD maintains high accuracy while enhancing inference, avoiding redundant processing during trajectory inference in existing methods. The feature optimization network employs the near-aware attention module (NAAM) to capture the pedestrian's latent motion patterns and interaction effects through key feature matching and local masking, learning spatiotemporal dependencies and generating the predicted trajectory encoding (PTE) with core information, thereby improving accuracy. The trajectory generation network predicts based on the PTE, avoiding further optimization and updates, significantly reducing computational burden, and enhancing inference speed. Experimental results demonstrate that NADP achieves state-of-the-art prediction accuracy and inference speed in both deterministic and stochastic trajectory prediction tasks on the ETH-UCY and SDD datasets.

## Requirements
- Python 3.8
- Tensorflow 2.4.0 (GPU)

## Models
The pre-trained models (ETH, HOTEL, UNIV, ZARA1, ZARA2) are saved in this repository, and all models can be acquired on <br />
[Models](https://drive.google.com/drive/folders/1I7eSd37ArGJt46ZfUSzXT0ciDvgW9m-K?usp=sharing) &nbsp; &nbsp; &nbsp;


## Usage

To (1) train the decoupled pedestrian trajectory prediction network model to obtain PTE and (2) validate the effectiveness of PTE for faster trajectory prediction on a specific dataset with a prediction network,  simply run the following command: 

```bash
# --dataset: ETH (default), HOTEL, UNIV, ZARA1, ZARA2, SDD
python train.py --dataset ETH
```
Please see ```train.py``` for more details.


To print evaluation results of the pedestrian trajectory prediction on the testing set, run:

```bash
# --dataset: ETH (default), HOTEL, UNIV, ZARA1, ZARA2, SDD
python evaluate.py --dataset ETH
```

Please see ```evaluate.py``` for more details.

## Results

| deterministic      |   ETH ↓   |  HOTEL ↓  |   UNIV ↓  |  ZARA1 ↓  |  ZARA2 ↓  | Average ↓ |
|--------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Social-GAN         | 1.13/2.21 | 1.01/2.18 | 0.60/1.28 | 0.42/0.91 | 0.52/1.11 | 0.67/1.41 |
| Trajectron++       | 1.02/2.09 | 0.33/0.63 | 0.52/1.16 | 0.42/0.94 | 0.32/0.71 | 0.52/1.11 |
| AgentFormer        | 1.67/3.65 | 1.61/3.59 | 1.62/3.53 | 1.85/4.13 | 1.68/3.74 | 1.69/3.73 |
| SocialVAE          | 0.97/1.93 | 0.40/0.78 | 0.54/1.16 | 0.44/0.97 | 0.33/0.74 | 0.54/1.12 |
| EigenTrajectory    | 0.93/2.05 | 0.33/0.64 | 0.58/1.23 | 0.45/0.99 | 0.34/0.75 | 0.53/1.13 |
| LMTraj-SUP         | 0.65/1.04 | 0.26/0.46 | 0.57/1.16 | 0.51/1.01 | 0.38/0.74 | 0.48/0.88 |
| HighGraph          | 0.62/1.14 | 0.62/1.28 | 0.51/1.25 | 0.32/0.61 | 0.29/0.75 | 0.47/1.01 |
| SingularTrajectory | 0.72/1.23 | 0.27/0.50 | 0.57/1.12 | 0.44/0.93 | 0.35/0.73 | 0.47/0.90 |
| [NADP (Ours)](https://github.com/123hzz/NADP) | 0.43/0.66 | 0.15/0.18 | 0.42/0.70 | 0.36/0.58 | 0.32/0.55 | 0.34/0.53 |


| stochastic         |   ETH ↓   |  HOTEL ↓  |   UNIV ↓  |  ZARA1 ↓  |  ZARA2 ↓  | Average ↓ |
|--------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| STAR               | 0.57/1.11 | 0.19/0.37 | 0.35/0.75 | 0.26/0.57 | 0.25/0.58 | 0.33/0.68 |
| AgentFormer        | 0.46/0.80 | 0.14/0.22 | 0.25/0.45 | 0.18/0.30 | 0.14/0.24 | 0.23/0.40 | 
| MID                | 0.57/0.93 | 0.21/0.33 | 0.29/0.55 | 0.28/0.50 | 0.20/0.37 | 0.31/0.54 |
| GP-Graph           | 0.43/0.63 | 0.18/0.30 | 0.24/0.42 | 0.17/0.31 | 0.15/0.29 | 0.23/0.39 |
| EqMotion           | 0.40/0.61 | 0.12/0.18 | 0.23/0.43 | 0.18/0.32 | 0.13/0.23 | 0.21/0.35 |
| EigenTrajectory    | 0.36/0.53 | 0.12/0.19 | 0.24/0.43 | 0.19/0.33 | 0.14/0.24 | 0.21/0.34 |
| LMTraj-SUP         | 0.41/0.51 | 0.12/0.16 | 0.22/0.34 | 0.20/0.32 | 0.17/0.27 | 0.22/0.32 |
| SingularTrajectory | 0.35/0.42 | 0.13/0.19 | 0.25/0.44 | 0.19/0.32 | 0.15/0.25 | 0.21/0.32 |
| [NADP (Ours)](https://github.com/123hzz/NADP) | 0.26/0.42 | 0.08/0.11 | 0.24/0.47 | 0.18/0.30 | 0.18/0.34 | 0.18/0.32 |


## Computational Complexity
| Methods    | # Accuracy | Training/Inference |
|------------|------------|--------------------|
| STAR       | 0.26/0.53  |  36.3h/97.0ms |
| AgentForme | 0.23/0.40  |  22h/8.2ms    |
| SocialVAE  | 0.21/0.33  |  2.1h/73.0ms  |
| LMTraj-SUP | 0.22/0.32  |  3.8h/18.3ms  |
| [NADP (Ours)](https://github.com/123hzz/NADP) | 0.19/0.33 | 3.5h/5.6ms |

## Citation
If you find this code useful for your research, please cite our paper
```bash

```

## License

NADP is released under the MIT License.