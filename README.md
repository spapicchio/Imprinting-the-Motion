# Overview of The Code
1. **BaseColab/MLDL_FPAR.pdf**: it contains the full explanation of our work
2. **Base Colab**: it contains the base colab used to perform all the training required for the project
3. **Script**: it contains the scripts used in Colab for calling the correct module
4. **module.py**: python files

# Imprinting The Motion
## Abstract
<i>First person action recognition (FPAR) task is one of the
most challenging in action recognition field. Most of the
existing works address this issue with two-stream architectures, where the visual appearance and the motion information of the object of interest, are exploited. In this paper, we
use as starting point the Ego-RNN architecture with the addition of the Motion Segmentation (MS) auxiliary task. We
propose the injection of a new branch in the architecture,
in order to employ the motion information more effectively.
This leads to have better predictions.</i>
## Our Architecture

<p align="center">
<img src="https://github.com/Simone-Papicchio/Imprinting-the-Motion/blob/main/images/architecture.jpg" alt="architeture" width="70%"/>
</p>
The Action Recognition Block extracts important spatial and temporal information from the video
with the exploitation of the ResNet-34 (mustard), Spatial Attention Layer (yellow) and ConvLSTM (orange). Moreover, it takes advantage
from the auxiliary task of the Motion Prediction Block, by embedding its knowledge inside the first layers of the backbone. This is
performed with a feedback branch (blue) that takes as input the features of the motion segmentation (MS) task (green). The Motion
Prediction Block takes, as input, the appearance features from the layer 4 of the ResNet and tries to identifies which parts of the image are
going to move

## Results

<p align="center">
<img src="https://github.com/Simone-Papicchio/Imprinting-the-Motion/blob/main/images/figure_1.jpg" alt="results" width="70%"/>
</p>

Our architecture is able to further exploit the motion information provided by the motion segmentation by merging them
with the appearance features in the first layers of the backbone.
The result is a model that better focuses on the relevant elements
for action recognition and this lead to the correct prediction (shake
tea cup instead of stir spoon cup)
