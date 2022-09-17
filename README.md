# DGMIL

On Updating.

Official PyTorch implementation of our MICCAI 2022 paper: DGMIL: Distribution Guided Multiple Instance Learning for Whole Slide Image Classification.

### Citation
If this work is helpful to you, please cite it as:
```
@InProceedings{10.1007/978-3-031-16434-7_3,
author="Qu, Linhao
and Luo, Xiaoyuan
and Liu, Shaolei
and Wang, Manning
and Song, Zhijian",
editor="Wang, Linwei
and Dou, Qi
and Fletcher, P. Thomas
and Speidel, Stefanie
and Li, Shuo",
title="DGMIL: Distribution Guided Multiple Instance Learning for Whole Slide Image Classification",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="24--34",
abstract="Multiple Instance Learning (MIL) is widely used in analyzing histopathological Whole Slide Images (WSIs). However, existing MIL methods do not explicitly model the data distribution, and instead they only learn a bag-level or instance-level decision boundary discriminatively by training a classifier. In this paper, we propose DGMIL: a feature distribution guided deep MIL framework for WSI classification and positive patch localization. Instead of designing complex discriminative network architectures, we reveal that the inherent feature distribution of histopathological image data can serve as a very effective guide for instance classification. We propose a cluster-conditioned feature distribution modeling method and a pseudo label-based iterative feature space refinement strategy so that in the final feature space the positive and negative instances can be easily separated. Experiments on the CAMELYON16 dataset and the TCGA Lung Cancer dataset show that our method achieves new SOTA for both global classification and positive patch localization tasks.",
isbn="978-3-031-16434-7"
}
```
