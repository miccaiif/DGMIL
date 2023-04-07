# :mushroom: DGMIL

This is a PyTorch/GPU implementation of our MICCAI 2022 paper [DGMIL: Distribution Guided Multiple Instance Learning for Whole Slide Image Classification](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_3#copyright-information).

Main models and training frameworks are uploaded. For patch generating, please follow [DSMIL](https://github.com/binli123/dsmil-wsi) for details. For MAE pretraining, please follow [MAE](https://github.com/facebookresearch/mae) for details.

<p align="center">
  <img src="https://github.com/miccaiif/DGMIL/blob/main/figure1.png" width="640">
</p>

### Frequently Asked Questions.

* The different result with [DSMIL](https://github.com/binli123/dsmil-wsi)


  Compared with DSMIL, for considerations of computational efficiency and resources, we used 5x (vs. DSMIL 20x) in our experiments. We used a patch size of 512 (vs DSMIL 224), and a patch is labeled as positive if it contains 25% or more cancer areas (not specified in DSMIL, please refer to its code). These different settings may result in the difference between the metrics reported by us and those reported by DSMIL.

### Citation
If this work is helpful to you, please cite it as:
```
@InProceedings{10.1007/978-3-031-16434-7_3,
author="Qu, Linhao
and Luo, Xiaoyuan
and Liu, Shaolei
and Wang, Manning
and Song, Zhijian",
title="DGMIL: Distribution Guided Multiple Instance Learning for Whole Slide Image Classification",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="24--34",
isbn="978-3-031-16434-7"
}
```

### Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).
