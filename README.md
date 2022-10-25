## Real-time head pose estimation and masked face classification for RGB facial image via knowledge distillation

This is a Pytorch implementation for the paper "Real-time head pose estimation and masked face classification for RGB facial image via knowledge distillation" published in Information Sciences 2022

Authors: Chien Thai, Viet Tran, Minh Bui, Dat Nguyen, Huong Ninh, and Hai Tran

[[PDF Version](https://drive.google.com/file/d/1kRrHh8G9z4u8YdqlUnlsm3doJZYQUfIC/view?usp=sharing)]
[[Modified Dataset](https://drive.google.com/drive/folders/19y6BCiMrV2GzD_v81_iRZ59E4IvcyD7h?usp=sharing)]
[[Pretrained Model](https://drive.google.com/file/d/1PQZeV-fkBX8U8X6NDIuc-c4AJ5M93Al3/view?usp=sharing)]

## Abstract

Recently, human head pose estimation and masked face classification are two essential problems in facial analysis. It is necessary to design a compact model to resolve both tasks in order to reduce the computational cost when deploying face recognition-based applications while maintaining accuracy. In this work, we proposed a lightweight multi-task model called MHPNet that simultaneously addresses both head pose estimation and masked face classification problems. Because of the lack of datasets with available labels for both tasks, we first train teacher models independently on two labelled datasets 300W-LPA and MAFA to extract the head pose and masked soft label. After that, we design architecture with ResNet18 backbone and two branches for two tasks and train our proposed model with the predictions of teacher models on joint datasets via the knowledge distillation process. To evaluate the effectiveness of our model, we use AFLW2000 and BIWI datasets for head pose estimation problems and MAFA datasets for masked face classification problems. Experiment results show that our proposed model significantly improves the accuracy compared to the state-of-the-art head pose estimation methods and achieves remarkable performance on masked face dataset. Furthermore, our model has the real-time speed of ∼400 FPS when inferring on a Tesla V100 GPU device.

## Experimental Results

![image](https://github.com/chientv99/maskpose/blob/main/headpose.png)

![image](https://github.com/chientv99/maskpose/blob/main/qualitative.png)

![image](https://github.com/chientv99/maskpose/blob/main/mask_result.png)

## Citation

```
@article{THAI2022,
title = {Real-time masked face classification and head pose estimation for RGB facial image via knowledge distillation},
journal = {Information Sciences},
year = {2022},
issn = {0020-0255},
author = {Chien Thai and Viet Tran and Minh Bui and Dat Nguyen and Huong Ninh and Hai Tran},
}
```
