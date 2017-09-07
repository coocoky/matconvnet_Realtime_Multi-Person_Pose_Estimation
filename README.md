# MatConvNet_Realtime_Multi-Person_Pose_Estimation

This is a matlab implementation of **Realtime Multi-Person Pose Estimation** using **matconvnet** backend, origin code is here <https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation>

## Introduction

Code repo for personal reproducing 2017 CVPR Oral paper using **MatConvNet**.

<p align="left">
<img src="https://github.com/ZheC/Multi-Person-Pose-Estimation/blob/master/readme/pose.gif", width="720">
</p>

This project is licensed under the terms of the [license](LICENSE).

**Disclaimer**

Copyright is owned by the [original author](Zhe Cao).

## Require

-  [MatConvNet](https://github.com/vlfeat/matconvnet)

```
cd <Pose_matconvnet>
git clone https://github.com/vlfeat/matconvnet.git
cd matconvnet
run matlab/vl_compilenn ;
```

- COCO Model (The model is freely available for free **non-commercial** use.)

  **Download** from [**google drive**](https://drive.google.com/open?id=0BwWEXCnRCqJ-MDNHMndYemU5bWc) or [**BaiduYun**](https://pan.baidu.com/s/1nuW8llR)

  or

  **Convert** caffe models to matconvnet by yourself.

  - Download caffemodel

    ```
    cd <Pose_matconvnet>
    cd models
    wget http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel
    ```

  - MATLAB

    ```
    cd models
    Converter_caffe_matconvnet
    build_openpose_net
    ```


## Testing

- MATLAB

```
cd <Pose_matconvnet>
cd testing
demo_matconvnet_pose();
```
## Related repository
- Our new C++ library [openPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release)
- [Pytorch version of the code](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)

## Citation
Please cite the paper in your publications if it helps your research:

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
      
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }