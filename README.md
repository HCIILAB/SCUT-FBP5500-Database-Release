# SCUT-FBP5500-Database-Release
A diverse benchmark database for multi-paradigm facial beauty prediction is now released by Human Computer Intelligent Interaction Lab of South China University of Technology. The database can be downloaded through the following link: (https://pan.baidu.com/s/1skLIxJ3  PASSWORD: jpqb)(Size = 167Mb). 

## Description
Facial beauty prediction is a significant visual recognition problem to make assessment of facial attractiveness
that is consistent to human perception. And the benchmark dataset is one of the most essential elements to achieve computation-based facial beauty prediction. Current datasets pertaining to facial beauty prediction are small and usually restricted to a very small and meticulously prepared subset of the population (e.g. ethnicity, gender and age).  
To tackle this problem, we build a new diverse benchmark dataset, called SCUT-FBP5500, to achieve multi-paradigm facial beauty prediction. 

The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties
(male/female, Asian/Caucasian, ages) and diverse labels (facial landmarks, beauty scores in 5 scales, beauty score distribution), which allows different computational model with different facial beauty prediction paradigms, such as appearance-based/shape-based facial beauty classification/regression/ranking model for male/female of Asian/Caucasian. 
Note: The SCUT-FBP5500 database can be only used for non-commercial research purpose. 
## Database Construction
The SCUT-FBP5500 Dataset can be divided into four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males. Most of the images of the SCUT-FBP5500 were collected from Internet, where some portions of Asian faces were from the DataTang, GuangZhouXiangSu and our laboratory, and some Caucasian faces were from the 10k US Adult Faces database.
### Beauty Scores and Facial Landmarks
All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers, and 86 facial landmarks are also  located to the significant facial components of each images. We developed a web-based GUI system to obtain the facial beauty scores and facial landmark locations respectively. 

### Training/Testing Set
We use two kinds of evaluation benchmarks to testing methods on SCUT-FBP5500, by spliting the SCUT-FBP5500 database in two ways: 
1.5-folds cross validation. For each cross validations, 4400 images are used for training and 1100 images are used for testing.  
2.6/4 training/testing. 

## Contact
Please consider to cite our paper when you use our database:


For any questions about this database please contact the authors by sending email to `lianwen.jin@gmail.com` and `lianglysky@gmail.com`.
