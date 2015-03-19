# Machine Learning Assignment- Deep learning at the gym

## Jakub Czakon


# Overview

The objective of this project was to predict in which way the subject exercises based on the data gather from ones mobile device. We have started our analysis by cleaning the data since, as can be easily found out the majority of the variables are in fact almost empty. That is why we discarded those variables and proceeded to classify on the fragment of the dataset. We used Principal component analysis to extract important features of the data and than Gradient Boosted Model ("gbm") for the classification. 

# Data Processing

The dataset that we used in the analysis consists of 160 columns or variables and 19622 records.However,as can be easily seen the majority of the variables are in fact almost empty so we decided to discard all variables that had less than 622 measured records. After that, and when we excluded dates and ids of the subjects we were left with the dataset consisting of 53 variables: 52 of which were dependend variables and one of them was the outcome. The following code was used.


```r
pml<-read.csv("pml-training.csv")

pml<-pml[,8:160]
pml<-apply(pml,2,function(x) as.character(x))
pml<-as.matrix(pml)
pml[(pml=="") | (pml=="#DIV/0!")]<-NA
pml<-as.data.frame(pml)

good<-sapply(pml, function(x) sum(is.na(x)))<19000
pml<-pml[,good]
pml_var<-apply(pml[,1:52],2,function(x) as.numeric(x))
pml_outcome<-data.frame(classe=as.factor(pml[,53]))
pml<-data.frame(pml_var,pml_outcome)
```

# Modeling

We divided our dataset to train and test.
We used Gradient Boosted Model ("gbm") for this project. To cope with overfitting or to get the sense of what is the out of sample error we used cross validation. We have also extracted features from the model using Principal component analysis to make the computations quicker. The following code was used:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pml_id<-createDataPartition(pml$classe,p=0.75,list=F)
pml_train<-pml[pml_id,]
pml_test<-pml[-pml_id,]

gbmFit <- train(classe ~ .,
                data = pml_train,
                method = c("gbm"),
                preProcess = c("pca"),
                trControl = trainControl(method = "cv", classProbs =  TRUE),
                verbose=FALSE
)
```

```
## Loading required package: gbm
```

```
## Warning: package 'gbm' was built under R version 3.1.3
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
```

After we have used our model to predict on the previuosly prepared test set the accurace was as following:

```r
confusionMatrix(predict(gbmFit,pml_test),pml_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1230  104   40   30   20
##          B   41  707   76   25   61
##          C   43   94  704   92   59
##          D   71   15   17  632   29
##          E   10   29   18   25  732
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8167          
##                  95% CI : (0.8056, 0.8274)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.768           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8817   0.7450   0.8234   0.7861   0.8124
## Specificity            0.9447   0.9487   0.9289   0.9678   0.9795
## Pos Pred Value         0.8638   0.7769   0.7097   0.8272   0.8993
## Neg Pred Value         0.9526   0.9394   0.9614   0.9585   0.9587
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2508   0.1442   0.1436   0.1289   0.1493
## Detection Prevalence   0.2904   0.1856   0.2023   0.1558   0.1660
## Balanced Accuracy      0.9132   0.8468   0.8761   0.8769   0.8960
```

Out of the sample error was calculated using the resampling sets. The accuracy measures for all the folds are presented below

```r
gbmFit$resample
```

```
##     Accuracy     Kappa Resample
## 1  0.8260870 0.7797816   Fold06
## 2  0.8152174 0.7657078   Fold09
## 3  0.8239293 0.7769810   Fold05
## 4  0.8383152 0.7951818   Fold08
## 5  0.8362772 0.7926439   Fold02
## 6  0.8340136 0.7900268   Fold04
## 7  0.8091033 0.7585658   Fold07
## 8  0.8104620 0.7600209   Fold10
## 9  0.8010862 0.7479401   Fold01
## 10 0.7982337 0.7443359   Fold03
```

That leaves us with the out of the sample error of 0.0145605 for the accuracy.
As it can be seen the predictive power is quite good and the out of the sample error reasonably strong leaving us with quite a good model.
