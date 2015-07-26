#Problem 1
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

# 1. Subset the data to a training set and testing set based on the Case variable in the data set. 
# 2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables 
#    and default caret settings. 
# 3. In the final model what would be the final model prediction for cases with the following variable values:
#   a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2 
# b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100 
# c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100 
# d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2 

which_train <- which(segmentationOriginal$Case == 'Train')
data.train <- segmentationOriginal[which_train,]
data.test <- segmentationOriginal[-which_train,]
set.seed(125)
model.fit <- train(Class ~ ., data=data.train, method="rpart")
plot(model.fit$finalModel, uniform=TRUE, main="Classification Tree")
text(model.fit$finalModel, pos=1, use.n=TRUE, all=TRUE, cex=.8)
install.packages(pkgs=c("rattle", "rpart.plot"), 
                 dependencies = c("Depends", "Imports"),
                 repos="http://cran.rstudio.com/");
library(rattle)
library(rpart)
library(rpart.plot)
fancyRpartPlot(model.fit$finalModel)
#a => PS
#b => WS
#c => PS
#d => ??

#question 3
# These data contain information on 572 different Italian olive oils from multiple regions in 
# Italy. Fit a classification tree where Area is the outcome variable. Then predict the value 
# of area for the following data frame using the tree command with all defaults
# 
# newdata = as.data.frame(t(colMeans(olive)))
# 
# What is the resulting prediction? Is the resulting prediction strange? Why or why not?
install.packages('pgmm')
library(pgmm)
data(olive)
olive = olive[,-1]
model.fit <- train(Area~., data=olive,  method="rpart")
model.fit <- rpart(Area~., data=olive,  method="class")
#getTree(model.fit$finalModel)
newdata = as.data.frame(t(colMeans(olive)))
model.fit=tree(newdata$Area ~ ., olive, subset=olive.train)
predict(model.fit, data=newdata)
packageVersion("pgmm")

#Question 4
install.packages("ElemStatLearn")
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

#Then set the seed to 13234 and fit a logistic regression model 
#(method="glm", be sure to specify family="binomial") with 
#Coronary Heart Disease (chd) as the outcome and age at onset, current alcohol 
#consumption, obesity levels, cumulative tabacco, type-A behavior, and low density lipoprotein 
#cholesterol as predictors. 
set.seed(13234)
model.fit <- train(chd ~ age+alcohol+obesity+tobacco+typea+ldl, data=trainSA, method="glm", family="binomial")

# Calculate the misclassification rate for your model using this function 
# and a prediction on the "response" scale:
#   
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

# What is the misclassification rate on the training set? 
# What is the misclassification rate on the test set?
missClass(trainSA$chd, predict(model.fit, newdata=trainSA))
missClass(testSA$chd, predict(model.fit, newdata=testSA))


#questino 5
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 
vowel.test$y <- as.factor(vowel.test$y)
vowel.train$y <- as.factor(vowel.train$y)
set.seed(33833)
model.fit <- train(y~., data=vowel.train, method="rf")
imp <- varImp(model.fit$finalModel)
imp[order(imp[1]),]
