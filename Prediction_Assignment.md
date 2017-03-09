# Prediction Assignment Writeup
Vadim K.  
2017-03-05  



# Synopsis
The goal of this project is to study the data from devices tracking physical activities of several enthusiasts and predict the manner in which they did the exercise.  
The data comes from http://groupware.les.inf.puc-rio.br/har in 2 files representing training and testing data sets.  
We will import and transform the data, choose variables, study several models and finish with building a prediction algorithm and use it to predict output on test data set. 

# Data Processing 

Loading the needed packages

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(e1071)
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

Downloading files and storing them in `data` folder (if not yet there)

```r
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!dir.exists("data")) {dir.create("data")}
if(!file.exists("data/pml-training.csv")) {
      download.file(url1, "data/pml-training.csv")
}
if(!file.exists("data/pml-testing.csv")) {
      download.file(url2, "data/pml-testing.csv")
}
```

Reading data

```r
training <- read.csv("data/pml-training.csv")
testing <- read.csv("data/pml-testing.csv")
```

Let's take a look at dimensions of our data set

```r
dim(training)
```

```
## [1] 19622   160
```

And the structure of the data (output omitted as it's too long)

```r
str(training)
```
we see that many variables contain some missing values.  
So next move is to remove from training set all variables with NA's

```r
training <- training[, colSums(is.na(training)) == 0]
```

Now let's identify and eliminate variables with variance close to zero, we'll use function `nearZeroVar` from caret package for this

```r
nZeroVar <- nearZeroVar(training)
training <- training[, -nZeroVar]
```

Let's take a look at the names of variables left in our data set

```r
names(training)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"          
##  [7] "roll_belt"            "pitch_belt"           "yaw_belt"            
## [10] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [13] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [16] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [19] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [22] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [25] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [28] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [31] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [34] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [37] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [40] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [43] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [46] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [49] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [52] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [55] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [58] "magnet_forearm_z"     "classe"
```

It looks like first 5 of them are not that useful (user name, time stamp, etc..) so we get rid of them

```r
training <- training[, -c(1:5)]
```

# Choosing the prediction algorithm

For further estimation of out-of-sample error we will divide the initial training set on two: sub-training and validation.

```r
seed <- 232323
set.seed(seed)
inTrain <- createDataPartition(training$classe, p = 3/4)[[1]]
sub.training <- training[inTrain, ]
validation <- training[-inTrain, ]
```

## Decision tree
First let's fit a model based on Decision tree method and build the prediction on validation set.

```r
set.seed(seed)
start_time <- proc.time()
fit_dt <- rpart(classe ~., data = sub.training)
finish_time <- proc.time() - start_time
print(finish_time)
```

```
##    user  system elapsed 
##    3.96    0.01    3.97
```
Model training took very little time. Let's check the accuracy on validation set

```r
pred_dt <- predict(fit_dt, newdata = validation, type = "class")
confusionMatrix(pred_dt, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1289  148   38   54   11
##          B   26  588   66   18   15
##          C   26   93  676   51   30
##          D   42  105   63  630  162
##          E   12   15   12   51  683
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7883          
##                  95% CI : (0.7766, 0.7997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7317          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9240   0.6196   0.7906   0.7836   0.7580
## Specificity            0.9285   0.9684   0.9506   0.9093   0.9775
## Pos Pred Value         0.8370   0.8247   0.7717   0.6287   0.8836
## Neg Pred Value         0.9685   0.9139   0.9556   0.9554   0.9472
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2628   0.1199   0.1378   0.1285   0.1393
## Detection Prevalence   0.3140   0.1454   0.1786   0.2043   0.1576
## Balanced Accuracy      0.9262   0.7940   0.8706   0.8464   0.8678
```
Accuracy of our prediction is 78.83% and the estimation of out-of sample error is 21%

```r
unname(1 - confusionMatrix(pred_dt, validation$classe)$overall[1])
```

```
## [1] 0.2116639
```
Not very impressive... Let's try something else.

## Support Vector Machines
We will continue with Support Vector Machines.  
This time we'll add K-fold cross-validation method with 3 folds.

```r
tr_control <- trainControl(method="cv", number = 3)

set.seed(seed)
start_time <- proc.time()
fit_svm <- svm(classe ~., data = sub.training, trControl = tr_control)
finish_time <- proc.time() - start_time
print(finish_time)
```

```
##    user  system elapsed 
##   82.68    0.27   84.67
```
There is more time spent on training, but let's check the accuracy

```r
pred_svm <- predict(fit_svm, newdata = validation)
confusionMatrix(pred_svm, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1381   53    2    3    0
##          B    5  864   24    0    2
##          C    8   26  819   72    5
##          D    1    1    9  728   28
##          E    0    5    1    1  866
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9498          
##                  95% CI : (0.9434, 0.9558)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9365          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9900   0.9104   0.9579   0.9055   0.9612
## Specificity            0.9835   0.9922   0.9726   0.9905   0.9983
## Pos Pred Value         0.9597   0.9654   0.8806   0.9492   0.9920
## Neg Pred Value         0.9960   0.9788   0.9909   0.9816   0.9913
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2816   0.1762   0.1670   0.1485   0.1766
## Detection Prevalence   0.2934   0.1825   0.1896   0.1564   0.1780
## Balanced Accuracy      0.9867   0.9513   0.9652   0.9480   0.9797
```
94.98% accuracy on validation set - not bad!  
We'll give another try with one more approach and then pick a winner.

## Random Forest
It's Random Forest's turn. We'll use the same cross-validation with it.

```r
set.seed(seed)
start_time <- proc.time()
fit_rf <- randomForest(classe ~., data = sub.training, trControl = tr_control)
finish_time <- proc.time() - start_time
print(finish_time)
```

```
##    user  system elapsed 
##   93.39    0.39   96.69
```
It took less time than for SVM. And what's the score?

```r
pred_rf <- predict(fit_rf, newdata = validation)
confusionMatrix(pred_rf, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  946    3    0    0
##          C    0    2  852    0    0
##          D    0    0    0  803    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9984          
##                  95% CI : (0.9968, 0.9993)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9979          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9965   0.9988   0.9989
## Specificity            0.9997   0.9992   0.9995   0.9998   0.9998
## Pos Pred Value         0.9993   0.9968   0.9977   0.9988   0.9989
## Neg Pred Value         1.0000   0.9992   0.9993   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1929   0.1737   0.1637   0.1835
## Detection Prevalence   0.2847   0.1935   0.1741   0.1639   0.1837
## Balanced Accuracy      0.9999   0.9980   0.9980   0.9993   0.9993
```
99.84% accuracy on validation set, thus estimation of out-of sample error is 0.16% - we have our winner.

# Applying the chosen algorithm
So now it's time to apply the chosen algorithm on our test data set and make a prediction that will finalize this assignment.

```r
predict(fit_rf, newdata = testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


