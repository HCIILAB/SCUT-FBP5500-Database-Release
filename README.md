# SCUT-FBP5500-Database-Release

A diverse benchmark database for multi-paradigm facial beauty prediction is now released by Human Computer Intelligent Interaction Lab of South China University of Technology. The database can be downloaded through the following link: (https://pan.baidu.com/s/1skLIxJ3  PASSWORD: jpqb) (Size = 167MB). 

## 1 Description

Facial beauty prediction is a significant visual recognition problem to make assessment of facial attractiveness
that is consistent to human perception. And the benchmark dataset is one of the most essential elements to achieve computation-based facial beauty prediction. Current datasets pertaining to facial beauty prediction are small and usually restricted to a very small and meticulously prepared subset of the population (e.g. ethnicity, gender and age). To tackle this problem, we build a new diverse benchmark dataset, called SCUT-FBP5500, to achieve multi-paradigm facial beauty prediction. 

The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties
(male/female, Asian/Caucasian, ages) and diverse labels (facial landmarks, beauty scores in 5 scales, beauty score distribution), which allows different computational model with different facial beauty prediction paradigms, such as appearance-based/shape-based facial beauty classification/regression/ranking model for male/female of Asian/Caucasian. 

Note: The SCUT-FBP5500 database can be only used for non-commercial research purpose. 

## 2 Database Construction

The SCUT-FBP5500 Dataset can be divided into four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males. Most of the images of the SCUT-FBP5500 were collected from Internet, where some portions of Asian faces were from the DataTang, GuangZhouXiangSu and our laboratory, and some Caucasian faces were from the 10k US Adult Faces database.
![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/SCUT-FBP5500.jpg)

All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers, and 86 facial landmarks are also  located to the significant facial components of each images. We developed a web-based GUI system to obtain the facial beauty scores and facial landmark locations respectively. 

### Training/Testing Set

We use two kinds of experiments settings to evaluate the facial beauty prediction methods on SCUT-FBP5500 benchmark, which includes: 

1) 5-folds cross validation. For each validation, 80% samples (4400 images) are used for training and the rest (1100 images) are used for testing.
2) The split of 60% training and 40% testing. 60% samples (3300 images) are used for training and the rest (2200 images) are used for testing.
We have provided the training and testing files in this link.  

## 3 Evaluation Results

We evaluate three different CNN models based on the structure of AlexNet,ResNet-18 and ResNeXt-50 on SCUT-FBP5500:

![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%205-folds%20cross%20validations.png)
![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%20the%20split%20of%2060%25%20training%20and%2040%25%20testing.png)

## 4 Citation and Contact

Please consider to cite our paper when you use our database:
```
@article{liang2017SCUT,
  title     = {SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
  author    = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
  jurnal    = {arXiv preprint },
  year      = {2017}
}
```

For any questions about this database please contact the authors by sending email to `lianwen.jin@gmail.com` and `lianglysky@gmail.com`.
