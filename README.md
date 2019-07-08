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

All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers, and 86 facial landmarks are also  located to the significant facial components of each images. Specifically, we save the facial landmarks in ‘pts’ format, which can be converted to 'txt' format using the code （e.g., pts2txt.py）. We developed several web-based GUI systems to obtain the facial beauty scores and facial landmark locations, respectively. 

### Training/Testing Set

We use two kinds of experimental settings to evaluate the facial beauty prediction methods on SCUT-FBP5500 benchmark, which includes: 

1) 5-folds cross validation. For each validation, 80% samples (4400 images) are used for training and the rest (1100 images) are used for testing.
2) The split of 60% training and 40% testing. 60% samples (3300 images) are used for training and the rest (2200 images) are used for testing.
We have provided the training and testing files in this link.  

## 3 Benchmark Evaluation

We evaluate three different CNN models (AlexNet, ResNet-18, ResNeXt-50) on SCUT-FBP5500 dataset for facial beauty prediction using two kinds of experimental settings, respectively. These CNNs are trained by initializing parameters with the models pretrained on ImageNet. Three different evaluation metrics are used in our experiments, including: Pearson correlation (PC), maximum absolute error (MAE), root mean square error (RMSE). The evaluation results are shown in the following, and more details are referred to our paper. 

![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%205-folds%20cross%20validations.png)
![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%20the%20split%20of%2060%25%20training%20and%2040%25%20testing.png) 

## 4 Resources of Trained Models

We release the codes of feed-forward implementation and several CNN models (like AlexNet, ResNet-18, ResNeXt-50) that were trained by the data of 'train_1.txt'. Please refer to the 'trained_models' folder for more details. The trained models can be downloaded through the following links: 
* Download link1 (faster for people in China): 
  https://pan.baidu.com/s/1byWe21ATKnpGarKY5feg1g> (password：owgm; decompression password: 12345)
* Download link2 (faster for people in other places): 
  https://drive.google.com/file/d/1un5CjTz_49Lg6MTNQn99WD7FjFqEJGoY/view (decompression password: 12345)


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
