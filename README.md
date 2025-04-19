# LncMamba

**LncMamba** is a machine learning project designed for predicting the subcellular localization of long non-coding RNAs (lncRNAs). It integrates several advanced components and provides an environment for training, testing, and visualizing the results of deep learning models. This project is based on Mamba2 and LncLocFormer, and it includes various modules for model building, training, and visualization.

## Requirements

To use this project, make sure you have the following installed:

- **Python**: Version 3.6 or above
- **numpy**: 1.22.3
- **scikit-learn**: 1.1.1
- **torch**: 1.13.1+cu117

Additionally, to use the selective scan with efficient hardware design, you need to install the `mamba_ssm` library with the following commands:

pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1



## Modules

### 1. **LncMamba**
   This module provides the base environment configuration and utilities for the project. It includes the core model definitions, which are key to the LncMamba framework.

   - **DL_ClassifierModel.py**: Contains the main deep learning model architecture for classifying the subcellular localization of lncRNAs.
   - **nnLayer.py**: Defines various neural network layers used in the model, ensuring modularity and flexibility in the design.

### 2. **Training and Visualization**
   - **train.py**: Contains the training loop for the model. After adjusting the dataset path, this script can be executed to start the training process.
   - **motif analysis.py**: Responsible for saving the model's weights after training and visualizing the results, including motif analysis.

Acknowledgement
This code is based on Mamba(https://github.com/state-spaces/mamba),LncLocFormer(https://github.com/CSUBioGroup/LncLocFormer). Thanks for their awesome work.

