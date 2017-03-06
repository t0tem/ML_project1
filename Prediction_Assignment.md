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

Downloading files and storing them in `data` folder (if not yet there)

```r
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
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

For further estimation of out-of-sample error we will divide the initial training set on two: sub-training and validation.

```r
set.seed(232323)
inTrain <- createDataPartition(training$classe, p = 3/4)[[1]]
sub.training <- training[inTrain,]
validation <- training[-inTrain,]
```



```r
str(sub.training)
```





