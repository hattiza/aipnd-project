"""
#### 1. Train

Train a new network on a data set with `train.py`

- Basic usage: 
    `python train.py data_directory`
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options: * Set directory to save checkpoints: 
    `python train.py data_dir --save_dir save_directory` 

* Choose architecture: 
    `python train.py data_dir --arch "vgg13"` 
* Set hyperparameters: 
    `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: 
    `python train.py data_dir --gpu`

"""

"""
RUBRIC

Training a network - train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint
Training validation log - The training loss, validation loss, and validation accuracy are printed out as a network trains
Model architecture - The training script allows users to choose from at least two different architectures available from torchvision.models
Model hyperparameters - The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
Training with GPU - The training script allows users to choose training the model on a GPU

"""


# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

#     Basic usage: python predict.py /path/to/image checkpoint
#     Options:
#         Return top K most likely classes: python predict.py ./flowers/test/1/image_06764.jpg ./vgg13_checkpoint.pth -k 3
#         Use a mapping of categories to real names: python predict.py ./flowers/test/1/image_06764.jpg ./vgg13_checkpoint.pth -cat cat_to_name.json -k 2
#         Use GPU for inference: python predict.py ./flowers/test/1/image_06764.jpg ./vgg13_checkpoint.pth -k 3 -cuda True

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset.')
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13', 'vgg16'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()


# feature_size = 25088 # for VGG16 model
# hidden_units = 4096
# unique_labels = 102
# learning_rate = 0.001

def main():

    args = parse_arguments()

    print(args)

if __name__ == '__main__':
    main()
