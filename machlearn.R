###################################
# CLEANING DATA
###################################
pml<-read.csv("pml-training.csv")
pml<-pml[,8:160]
pml<-apply(pml,2,function(x) as.character(x))
pml<-as.matrix(pml)
pml[(pml=="") | (pml=="#DIV/0!")]<-NA
pml<-as.data.frame(pml)

sapply(pml, function(x) sum(is.na(x)))

good<-sapply(pml, function(x) sum(is.na(x)))<19000
pml<-pml[,good]
pml_var<-apply(pml[,1:52],2,function(x) as.numeric(x))
pml_outcome<-data.frame(classe=as.factor(pml[,53]))
pml<-data.frame(pml_var,pml_outcome)

###################################
# MODELING
###################################

library(caret)

pml_id<-createDataPartition(pml$classe,p=0.75,list=F)
pml_train<-pml[pml_id,]
pml_test<-pml[-pml_id,]

gbmFit <- train(classe ~ .,
                data = pml_train,
                method = c("gbm"),
                preProcess = c("pca"),
                trControl = trainControl(method = "cv", classProbs =  TRUE)
)


confusionMatrix(predict(gbmFit,pml_test),pml_test$classe)
gbmFit$resample

###################################
# testing for submission
###################################

submision_test<-read.csv("pml-testing.csv")
submision_test<-submision_test[8:160]
used_var_id<-colnames(submision_test) %in% colnames(pml_var)
submision_test<-submision_test[,used_var_id]

predict(gbmFit,submision_test)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

answers<-predict(gbmFit,submision_test)
pml_write_files(answers)
