# :mushroom: DGMIL

This is a PyTorch/GPU implementation of our MICCAI 2022 paper [DGMIL: Distribution Guided Multiple Instance Learning for Whole Slide Image Classification](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_3#copyright-information).

Main models and training frameworks are uploaded. For patch generating, please follow [DSMIL](https://github.com/binli123/dsmil-wsi) for details. For MAE pretraining, please follow [MAE](https://github.com/facebookresearch/mae) for details.

<p align="center">
  <img src="https://github.com/miccaiif/DGMIL/blob/main/figure1.png" width="640">
</p>

### Frequently Asked Questions.

* The different result with [DSMIL](https://github.com/binli123/dsmil-wsi)


  Compared with DSMIL, for considerations of computational efficiency and resources, we used 5x (vs. DSMIL 20x) in our experiments. We used a patch size of 512 (vs DSMIL 224), and a patch is labeled as positive if it contains 25% or more cancer areas (not specified in DSMIL, please refer to its code). These different settings may result in the difference between the metrics reported by us and those reported by DSMIL.

* Regarding the issues with the MAE model in the article.

  As mentioned in the article, we used the official MAE model to pre-train a feature extractor for pre-extracting features of all patches (since the number of patches cut from WSI is too large, direct end-to-end training based on the RGB image is too costly, and the existing common methods all adopt pre-trained feature extractors for feature extraction). Subsequently, we only trained a simple projector (inspired by [SimCLR](https://github.com/sthalles/SimCLR) to complete the mapping of the feature space, while the pre-extracted features remain unchanged. This can also be understood as the overall feature extractor including a pre-trained but fixed MAE model during training, and a projector that is updated during training.

  For specific implementations, as the settings of different MIL experiments vary (such as patch size, scale, etc.), patching needs to be conducted according to your own experimental settings. The [DSMIL](https://github.com/binli123/dsmil-wsi) paper provides a good example for reference (and is also referenced in this article). For MAE self-supervised feature extraction, you can refer to [MAE](https://github.com/facebookresearch/mae)'s official code, and many pre-trained models are available for use. In this paper, an MAE model was trained on the dataset after patching using the official MAE code.

  As uploading all these extracted feats files would require a lot of time and space, we have open-sourced the main and key code models. The guidance in the readme and the above guidance can support the reproduction of this work. Thank you again for your attention! You are welcome to contact and cite us! Thank you!


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
