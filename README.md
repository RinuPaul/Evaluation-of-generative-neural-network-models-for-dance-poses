# Evaluation-of-generative-neural-network-models-for-dance-poses

This repository contains all the codes related to the thesis topic 'Evaluation-of-generative-neural-network-models-for-dance-poses'.

# Abstract:
The generation of dance synchronized with music is challenging in nature because of the complex characteristics of dance and music features. Recently, different models have been introduced to address human motion, and especially dance motion. In this thesis we compare four state-of-the-art probabilistic neural network models for the generation of dance motion in accordance with music: Human Prediction Generative Adversarial Network (HPGAN), Motion Variational Auto-Encoder (MVAE), Dance Revolution and TransFlower. The models are evaluated for their quality and variation. The Laban Movement Analysis (LMA) is used to measure the naturalness with respect to human-like dance characteristics in comparison to the ground truth. To understand their hardware restrictions the models are also compared for efficiency in terms of memory usage and inference speed. The possibilities of deploying these models directly in Unity and on Virtual Reality (VR) hardware are also explored. 

The preprocessing of the AIST++ datset to convert the dance clips into point cloud data representation and the extraction of audio features are given in data_preprocessing_for_ref_models.py.

The LMA evaluation codes are Dance_revflow-point_cloud/lma_dance.py, Dance_revflow-point_cloud/correlation_dance.py for the evaluation of the generated dance sequences of the models.

The visualization of the generated dance sequence is done using client-server method in Unity. The dance clips generated in Python (server) is displayed in Unity (client). The repository used for this purpose is : https://github.com/eherr/mi_variational_dance_motion_vis. 

Reference for the AIST++ database:

R. Li, S. Yang, D. A. Ross, and A. Kanazawa, “Learn to dance with AIST++: music conditioned 3d dance generation, ”arxiv preprint arxiv:2101.08779, 2021.

