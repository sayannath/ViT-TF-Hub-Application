![GitHub forks](https://img.shields.io/github/forks/sayannath/ViT-TF-Hub-Application?style=for-the-badge)
![GitHub Repo stars](https://img.shields.io/github/stars/sayannath/ViT-TF-Hub-Application?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/sayannath/ViT-TF-Hub-Application?style=for-the-badge)
![Twitter Follow](https://img.shields.io/twitter/follow/sayannath2350?style=for-the-badge)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg?style=for-the-badge)](https://gitHub.com/sayannath)

# Vision Transformer TF-Hub Application

## Description
This repositories show how to `fine-tune` a Vision Transformer model from [TensorFlow Hub](https://www.tfhub.dev) on the Image Scene Detection dataset.

## Dataset Used
A newly collected Camera Scene Classification dataset consisting of images belonging to 30 different classes. This dataset is the part of the competition which is [Mobile AI Workshop @ CVPR 2021](https://competitions.codalab.org/competitions/28113).
You can find the dataset details [here](https://competitions.codalab.org/competitions/28113#participate).

## Models

These models are available on [TensorFlow Hub](https://www.tfhub.dev) for Vision Transformer.

### Image Classifiers

* [ViT-S16](https://tfhub.dev/sayakpaul/vit_s16_classification/1)
* [ViT-B8](https://tfhub.dev/sayakpaul/vit_b8_classification/1)
* [ViT-B16](https://tfhub.dev/sayakpaul/vit_b16_classification/1)
* [ViT-B32](https://tfhub.dev/sayakpaul/vit_b32_classification/1)
* [ViT-L16](https://tfhub.dev/sayakpaul/vit_l16_classification/1)
* [ViT-R26-S32 (light augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_lightaug_classification/1)
* [ViT-R26-S32 (medium augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_medaug_classification/1)
* [ViT-R50-L32](https://tfhub.dev/sayakpaul/vit_r50_l32_classification/1)

### Feature Extractors

* [ViT-S16](https://tfhub.dev/sayakpaul/vit_s16_fe/1)
* [ViT-B8](https://tfhub.dev/sayakpaul/vit_b8_fe/1)
* [ViT-B16](https://tfhub.dev/sayakpaul/vit_b16_fe/1)
* [ViT-B32](https://tfhub.dev/sayakpaul/vit_b32_fe/1)
* [ViT-L16](https://tfhub.dev/sayakpaul/vit_l16_fe/1)
* [ViT-R26-S32 (light augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_lightaug_fe/1)
* [ViT-R26-S32 (medium augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_medaug_fe/1)
* [ViT-R50-L32](https://tfhub.dev/sayakpaul/vit_r50_l32_fe/1)

> Note: As we want to fine-tune our model so we used the feature-extractor model and build the image classifier.

## Benchmark Results

| Sl No | Models                   | No of Parameters | Accuracy | Loss   | Validation Accuracy | Validation Loss |
|-------|--------------------------|------------------|----------|--------|---------------------|-----------------|
| 1     | ViT-S/16                 | 21,677,214       | 99.73%   | 1.72%  | 96.87%              | 13.39%          |
| 2     | ViT R26-S/32(light aug)  | 36,058,462       | 99.70%   | 1.38%  | 96.67%              | 14.38%          |
| 3     | ViT R26-S/32(medium aug) | 36,058,462       | 99.80%   | 1.15%  | 97.17%              | 14.50%          |
| 4     | ViT B/32                 | 87,478,302       | 99.43%   | 2.76%  | 96.87%              | 10.63%          |
| 5     | MobileNetV3Small         | 2,070,158        | 95.20%   | 22.87% | 92.73%              | 21.49%          |
| 6     | MobileNetV2              | 2,929,246        | 95.06%   | 22.35% | 88.89%              | 42.24%          |
| 7     | BigTransfer (BiT)        |                  | 99.53%   | 3.41%  | 96.97%              | 9.49%           |

> Note: Last three results are benchmarked during thr CVPR Competition. You can find the repository [here](https://github.com/sayannath/Image-Scene-Classification).

## Notebooks
 - [x] ViT S/16 
 - [x] ViT R26-S/32(Light Augmentation) 
 - [x] ViT R26-S/32(Medium Augmentation)
 - [x] ViT B/32 
 - [ ] ViT R50-L/32
 - [ ] ViT B/16
 - [ ] ViT L/16
 - [ ] ViT B/8

## Links
| Sl No | Models                   | Colab Notebook | TensorBoard |
|----|--------------------------|----------------|-------------|
| 1  | ViT-S/16                 | [Link](https://colab.research.google.com/drive/1ISB3E5_wjojRjhbCjRLaKLCPxUHqtxd1?usp=sharing)       | [Link](https://tensorboard.dev/experiment/m9OMnYIzTw66LWXvyXCYgg/)    |
| 2  | ViT R26-S/32(light aug)  | [Link](https://colab.research.google.com/drive/14Ms__eAJOD0jdDLlHxmIawcQET_GyQjz?usp=sharing)       | [Link](https://tensorboard.dev/experiment/myd5IEZtRjWEmAQQ9lSolA/)    |
| 3  | ViT R26-S/32(medium aug) | [Link](https://colab.research.google.com/drive/1xuQTvl5lYqR3tn_17d7_WeDrdj76ieIl?usp=sharing)       | [Link](https://tensorboard.dev/experiment/35bwOLWxQLqO0E11sdveDQ/)    |
| 4  | ViT B/32                 | [Link](https://colab.research.google.com/drive/1-9mo1H8tOHOjqunF317a-I1B4vbX6yeC?usp=sharing)       | [Link](https://tensorboard.dev/experiment/H2QSxurmQt6YNVVWSlTaUA/)    |

> Each directory of model contains the particular notebook, python script, metric graph, train-logs(in .csv) and TensorBoard callbacks.

## References

[1] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)

[2] [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers by Steiner et al.](https://arxiv.org/abs/2106.10270)

[3] [Vision Transformer GitHub](https://github.com/google-research/vision_transformer)

[4] [jax2tf tool](https://github.com/google/jax/tree/main/jax/experimental/jax2tf/)

[5] [Image Classification with Vision Transformer in Keras](https://keras.io/examples/vision/image_classification_with_vision_transformer/)

[6] [ViT-jax2tf](https://github.com/sayakpaul/ViT-jax2tf)

[7] [Vision Transformers are Robust Learners](https://arxiv.org/abs/2105.07581), [Repository](https://github.com/sayakpaul/robustness-vit)

[8] [Vision Transformer TF-Hub Model Collection](https://tfhub.dev/sayakpaul/collections/vision_transformer/1)

## Acknowledgements

* Thanks to [Sayak Paul](https://sayak.dev) for building the models of ViT so that we can use Vision Transformer in a straight way.
* Thanks to the authors of Vision Transformers for their efforts put into open-sourcing the models.

## Contributors

<a href="https://github.com/sayannath/ViT-TF-Hub-Application/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=sayannath/ViT-TF-Hub-Application" />
</a>

