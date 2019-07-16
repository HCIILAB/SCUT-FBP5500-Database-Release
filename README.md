# SCUT-FBP5500-Database-Release

A diverse benchmark database (Size = 172MB) for multi-paradigm facial beauty prediction is now released by Human Computer Intelligent Interaction Lab of South China University of Technology. The database can be downloaded through the following links: 
* Download link1 (faster for people in China): 

  https://pan.baidu.com/s/1Ff2W2VLJ1ZbWSeV5JbF0Iw  (PASSWORD: if7p) 
* Download link2 (faster for people in other places): 

  https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf

## 1 Description

The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties
(male/female, Asian/Caucasian, ages) and diverse labels (facial landmarks, beauty scores in 5 scales, beauty score distribution), which allows different computational model with different facial beauty prediction paradigms, such as appearance-based/shape-based facial beauty classification/regression/ranking model for male/female of Asian/Caucasian. 

## 2 Database Construction

The SCUT-FBP5500 Dataset can be divided into four subsets with different races and gender, including 2000 Asian females(AF), 2000 Asian males(AM), 750 Caucasian females(CF) and 750 Caucasian males(CM). Most of the images of the SCUT-FBP5500 were collected from Internet, where some portions of Asian faces were from the DataTang, GuangZhouXiangSu and our laboratory, and some Caucasian faces were from the 10k US Adult Faces database.
![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/SCUT-FBP5500.jpg)

All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers, and 86 facial landmarks are also  located to the significant facial components of each images. Specifically, we save the facial landmarks in ‘pts’ format, which can be converted to 'txt' format by running pts2txt.py. We developed several web-based GUI systems to obtain the facial beauty scores and facial landmark locations, respectively. 

### Training/Testing Set Split

We use two kinds of experimental settings to evaluate the facial beauty prediction methods on SCUT-FBP5500 benchmark, which includes: 

1) 5-folds cross validation. For each validation, 80% samples (4400 images) are used for training and the rest (1100 images) are used for testing.
2) The split of 60% training and 40% testing. 60% samples (3300 images) are used for training and the rest (2200 images) are used for testing.
We have provided the training and testing files in this link.  

## 3 Training Tutorials and Models

We trained three different CNN models (AlexNet, ResNet-18, ResNeXt-50) on SCUT-FBP5500 dataset for facial beauty prediction by using the L2-norm distance loss. Each raw RGB image is resized as 256\*256, and then a 227\*227 random crop of raw image is obtained to feed into AlexNet, while a 224\*224 random crop is sent to ResNet and ResNeXt. The model parameters are initialized by the pretrained CNN models of ImageNet and updated by mini-batch Stochastic Gardient Descent (SGD), where the learning rate is initialized as 0.001 and decreased by a factor of 10 per 5000 iterations. We set the batchsize as 16, momentum coefficient as 0.9, maximum iterations as 20000, and weight decay coefficient as 5e-4 for AlexNet while 1e-4 for ResNet and ResNeXt.

All the experiments were implemented on two different platforms separately, Caffe and Pytorch. And we release the codes of feed-forward implementation and the CNN models that were trained by the data of 'train_1.txt'. Please refer to the 'trained_models_for_caffe' and 'trained_models_for_pytorch' folders for more details. 
### Trained Models for Caffe
The trained models for Caffe (Size = 322MB) can be downloaded through the following links: 
* Download link1 (faster for people in China): 

  https://pan.baidu.com/s/1byWe21ATKnpGarKY5feg1g> (PASSWORD：owgm; Zip PASSWORD: 12345)
* Download link2 (faster for people in other places): 

  https://drive.google.com/file/d/1un5CjTz_49Lg6MTNQn99WD7FjFqEJGoY/view (Zip PASSWORD: 12345)
#### Requirements:
* Python 2.7
* Caffe
* Numpy
* Matplotlib
* Scikit-image

### Trained Models for Pytorch
And the trained models for Pytorch (Size = 101MB) can be downloaded throught the following link:
* Download link: 

https://pan.baidu.com/s/1OhyJsCMfAdeo8kIZd29yAw (PASSWORD: ateu)
#### Requirements:
* Python 2.7
* Torch 1.0.1
* Numpy
* Pillow

## 4 Benchmark Evaluation

We set AlexNet, ResNet-18, and ResNeXt-50 as the benchmarks of the SCUT-FBP5500 dataset, and we evaluate the benchmark on various measurement metrics, including: Pearson correlation (PC), maximum absolute error (MAE), and root mean square error (RMSE). The evaluation results are shown in the following. Please refer to our paper for more details. 

![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%205-folds%20cross%20validations.png)
![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%20the%20split%20of%2060%25%20training%20and%2040%25%20testing.png) 


## 5 Citation and Contact

Please consider to cite our paper when you use our database:
```
@article{liang2017SCUT,
  title     = {SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
  author    = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
  jurnal    = {ICPR},
  year      = {2018}
}
```

Note: The SCUT-FBP5500 database can be only used for non-commercial research purpose. 

For any questions about this database please contact the authors by sending email to `lianwen.jin@gmail.com` and `lianglysky@gmail.com`.


##  Desclaimer

This AI algorithm is purely for academic research purpose. The dataset and codes are for academic research use only. We are not responsible for the objectivity and accuracy of the proposed model and algorithm.
