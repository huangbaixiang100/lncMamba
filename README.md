# LncLocFormer

**LncLocFormer** is a machine learning project designed for predicting the subcellular localization of long non-coding RNAs (lncRNAs). It integrates several advanced components and provides an environment for training, testing, and visualizing the results of deep learning models. This project is based on Mamba2, and it includes various modules for model building, training, and visualization.

## Requirements

- **Python**: Version 3.6 or above
- **Mamba2**: For managing the environment
- **PyTorch**: For model training and inference
- **Matplotlib**: For visualizing the results
- **Other dependencies**: All dependencies can be installed through the provided environment configuration.

## Modules

### 1. **LncLocFormer**
   This module provides the base environment configuration and utilities for the project. It includes the core model definitions, which are key to the LncLocFormer framework.

   - **DL_ClassifierModel.py**: Contains the main deep learning model architecture for classifying the subcellular localization of lncRNAs.
   - **nnLayer.py**: Defines various neural network layers used in the model, ensuring modularity and flexibility in the design.

### 2. **Training and Visualization**
   - **Training_Code.py**: Contains the training loop for the model. After adjusting the dataset path, this script can be executed to start the training process.
   - **Save_Model_Weights_and_Visualization.py**: Responsible for saving the model's weights after training and visualizing the results, including training and validation curves.

### 3. **Mamba**
   The Mamba module provides the official Mamba model code and supports the invocation and execution of the `KGPDPLAM_alpha_Mamba2` within the `DL_ClassifierModel.py`.

