if (!require("caret") || !require("doParallel")) {
  install.packages(pkgs = c("doParallel", "carat"), 
                   dependencies = c("Depends", "Imports"),
                   repos="http://cran.rstudio.com/")
  library("caret")
  library("doParallel")
}

#setwd('~/Desktop/machine_learning/assignment/')
#allow for multithreading
registerDoParallel(cores=3)


#HELPER FUNCTIONS

#' @param dataSet - a dataframe of factors
#' @param factors - a vector of substrings to
#' @desc iterate over the substrings and remove the factors
#' that contain substrings in their headings
removeFactorSets <- function(dataSet, factors) {
  for(factor in factors) {
    columns <- dataSet[,grepl(factor, names(dataSet))]
    dataSet <- dataSet[!names(dataSet) %in% names(columns)]
  }
  return(dataSet)
}

#' @param dataSet - a dataframe of factors
#' @param factors - a vector of substrings to
#' @desc iterate over the substrings and convert the columns to numbers
convertFactorToNumeric <- function(dataFrame, factors) {
  for(factor in factors) {
    columns <- dataFrame[,substr(colnames(dataFrame),1,nchar(factor))==factor]
    for(heading in names(columns)){
      dataFrame[heading] <- as.numeric(levels(dataFrame[heading]))[dataFrame[heading]];
    }
  }
  return(dataFrame)
}

#' @param dataSet - a dataframe of factors
#' @param threshold - the threshold 
#' @desc find factors and remove factors that have a ratio of invalid values 
findNoisyFactors <- function (dataFrame, threshold) {
  dataFrame.nrow <- nrow(dataFrame)
  # Run apply on the training set.  Dimension 2 means call the function on each column.
  col_na_which <- apply(dataFrame, 2, function(col) {
    sum(is.na(col)) / dataFrame.nrow > threshold
  })
  return(names(dataFrame[col_na_which]))
}

####################################
# Generate the model


## HELPERS
                             
 #' @param modelRda - a model data file
 #' @param noiseTolerance - Amount of noise to tolerate before discarding factor 
 #' @param trainingRatio - size of training set relative to testing
 #' @param trainingRatio - factors to be excluded
 cleanData <- function (m, noiseTolerance = 0.7
                        , trainingRatio = 0.8
                        , exclusionSet = c("timestamp"
                                           #,"kurtosis"
                                           #,"skewness" 
                        )
                        , exclusions = NULL) {
  
  if(is.null(exclusions))
    exclusions <- c("X", 
                    "user_name", #not sensor data
                    "new_window", #not sensor data
                    "num_window", #not sensor data
                    "amplitude_yaw_belt", #garbage data
                    "amplitude_yaw_dumbbell", #garbage data
                    "amplitude_yaw_forearm",#garbage data
                    "cvtd_timestamp", #not correlated
                    "skewness_yaw_forearm", #garbage data
                    "kurtosis_yaw_forearm", #garbage data
                    "skewness_yaw_dumbbell", #garbage data
                    "kurtosis_yaw_dumbbell", #garbage data
                    "skewness_yaw_belt", #garbage data
                    "kurtosis_yaw_belt"); #garbage data
  
    
  ###FEATURES PREPROCESSING
  #come up with a set of the relevant factors (do some exploratory analysis to exclude varaibles)
  
  inTrain <- createDataPartition(dataTrainPml$classe, p=trainingRatio, list=FALSE)
  training <- dataTrainPml[inTrain,]; testing <- dataTrainPml[-inTrain,]
  
  #create set of garbage factors
  noisy <- findNoisyFactors(training, noiseTolerance)
  exclusions <- unique(c(exclusions, noisy));
  #additional sets of invalid factors to find and remove by string matching
  
  #remove garbage factors
  trainPre <- training;
  trainPre <- trainPre[,!names(trainPre) %in% exclusions];
  trainPre <- removeFactorSets(trainPre, exclusionSet);
  
  #convert types
  trainPre$classe <- as.factor(trainPre$classe)
  if("user_name" %in% names(trainPre))
    trainPre$user_name <- as.factor(trainPre$user_name)
  #kurtosis, skewedness, and yaw are factors but should be values
  #trainPre <- convertFactorToNumeric(trainPre, c("kurtosis", "skewedness"))
  
  
  ####do some exploratory analysis
  #nearZeroVar(training, saveMetrics=TRUE)
  #dummies <- dummyVars(user_name ~ classe, data=training)
  #head(predict(dummies, newdata=training))
  
  
  ###ALGORITHM & PARAMETERS
  
  #principle components analysis
  # M <- abs(cor(trainPre[,-ncol(trainPre)]))
  # diag(M) <- 0 
  # pca.factors <- which(M > 0.95, arr.ind = T)
  # nrow(pca.factors)
  # trainPC <- prcomp(trainPre[,c(2,5,11,3,9,32,34)])
  # 
  # trainProc <- preProcess(trainPre[,-ncol(trainPre)], method="pca", pcaComp = 38)
  # trainPC <- predict(trainProc, trainPre[,-ncol(trainPre)])
  
  #graph the PCA
  #plot(trainPC[,1],trainPC[,2], col=trainPre$classe)
  
  ##apply transforms to testing
  testing$classe <- as.factor(testing$classe)
  testing$user_name <- as.factor(testing$user_name)
  return(list(training=trainPre, testing=testing))
}
  
#' @param modelRda - a model data file
#' @param noiseTolerance - Amount of noise to tolerate before discarding factor 
#' @param trainingRatio - size of training set relative to testing
#' @param trainingRatio - factors to be excluded
generateModel <- function (modelRds = 'modelRF_80train_names_windows_70noise.Rdata', training, testing) {
  
  if(file.exists(modelRds)){
    model <- readRDS(file=modelRds)
  } else {
    #random forest
    model <- train(classe ~ ., method="rf", data=training, 
                     #trControl=trainControl(method="cv",number=5),
                     prox=TRUE,allowParallel=TRUE)
    saveRDS(model, file=modelRds)
  }
  
  #modelGLM <- train(trainPre$classe ~ ., method="glm", data=trainPC)
  
  
  ###CROSS VALIDATION
  #Predict the classe variable 
  
  #perform transformations
  #testPre <- testing
  #testPre <- testPre[,!names(testPre) %in% exclusions]
  #testPre <- removeFactorSets(testPre, exclusion_sets)

  #evaluate random forest model
  cm <- confusionMatrix(testing$classe, predict(model, testing))
  
  #evaluate evaluate glm with PCA
#   testPC <- predict(trainProc, testing[,-ncol(testing)])
#   confusionMatrix(testing$classe, predict(modelGLM, testPC))
  
  return(list(model=model, confusionMatrix=cm))
}


###BEGIN EXERCISE

#load data from disk
dataTrainPml <- read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings=c("NA", "#DIV/0!"))
dataTestingPml <- read.csv("pml-testing.csv", stringsAsFactors=FALSE, na.strings=c("NA", "#DIV/0!"))

#general preprocessing
dataTestingPml$user_name <- as.factor(dataTestingPml$user_name)

#EVALUATION
# gyro <- which(grepl("^gyro", colnames(data1$training), ignore.case = F))
# gyros <- data1$training[, gyro]
# 
# featurePlot(x = gyros, y = data1$training$classe, pch = 19, main = "Feature plot", 
#             plot = "pairs")

# set.seed(1242)
# modelFile <- 'modelRF_80train_names_windows_70noise.Rds'
# data1 <- cleanData(noiseTolerance = 0.8
#                   , trainingRatio = 0.80)
# result1 <- generateModel(modelRds = modelFile, data1$training, data1$testing)
# 
# result1$confusionMatrix
# predict(result1$model, dataTestingPml)
# 
# set.seed(1242)
# modelFile <- 'modelRF_80train_names_windows_70noise.Rds'
# data2 <- cleanData(noiseTolerance = 0.7
#                   , trainingRatio = 0.8
#                   , exclusionSet = c("timestamp"
#                                      #,"kurtosis"
#                                      #,"skewness" 
#                                     )
#                   , exclusions = c("X", 
#                                    #"user_name", #not sensor data
#                                    #"new_window", #not sensor data
#                                    #"num_window", #not sensor data
#                                    "amplitude_yaw_belt", #garbage data
#                                    "amplitude_yaw_dumbbell", #garbage data
#                                    "amplitude_yaw_forearm",#garbage data
#                                    "cvtd_timestamp", #not correlated
#                                    "skewness_yaw_forearm", #garbage data
#                                    "kurtosis_yaw_forearm", #garbage data
#                                    "skewness_yaw_dumbbell", #garbage data
#                                    "kurtosis_yaw_dumbbell", #garbage data
#                                    "skewness_yaw_belt", #garbage data
#                                    "kurtosis_yaw_belt") #garbage data
#                     )
# result2 <- generateModel(modelRds = modelFile
#                        , training = data2$training
#                        , testing = data2$testing )
# result2$confusionMatrix
# predict(result2$model, dataTestingPml)
# 
# set.seed(1242)
# modelFile <- 'modelRF_75train_notimestamps.Rds'
# data3 <- cleanData(noiseTolerance = 0.5
#                   , trainingRatio = 0.75
#                   , exclusionSet = c("timestamp"
#                                      #,"kurtosis"
#                                      #,"skewness" 
#                   )
#                   , exclusions = c("X", 
#                                    "user_name", #not sensor data
#                                    "new_window", #not sensor data
#                                    "num_window", #not sensor data
#                                    "amplitude_yaw_belt", #garbage data
#                                    "amplitude_yaw_dumbbell", #garbage data
#                                    "amplitude_yaw_forearm",#garbage data
#                                    "cvtd_timestamp", #not correlated
#                                    "skewness_yaw_forearm", #garbage data
#                                    "kurtosis_yaw_forearm", #garbage data
#                                    "skewness_yaw_dumbbell", #garbage data
#                                    "kurtosis_yaw_dumbbell", #garbage data
#                                    "skewness_yaw_belt", #garbage data
#                                    "kurtosis_yaw_belt") #garbage data
#                   )
# result3 <- generateModel(modelRds = modelFile
#                        , training = data3$training
#                        , testing = data3$testing )
#                        
# result3$confusionMatrix
# predict(result3$model, dataTestingPml)
# 
# set.seed(1242)
# modelFile <- 'modelRF_75train_notimestamps_50noise.Rds'
# data4 <- cleanData(noiseTolerance = 0.5
#                   , trainingRatio = 0.75
#                   , exclusionSet = c("timestamp"
#                                      #,"kurtosis"
#                                      #,"skewness" 
#                   )
#                   , exclusions = c("X", 
#                                    "user_name", #not sensor data
#                                    "new_window", #not sensor data
#                                    "num_window", #not sensor data
#                                    "amplitude_yaw_belt", #garbage data
#                                    "amplitude_yaw_dumbbell", #garbage data
#                                    "amplitude_yaw_forearm",#garbage data
#                                    "cvtd_timestamp", #not correlated
#                                    "skewness_yaw_forearm", #garbage data
#                                    "kurtosis_yaw_forearm", #garbage data
#                                    "skewness_yaw_dumbbell", #garbage data
#                                    "kurtosis_yaw_dumbbell", #garbage data
#                                    "skewness_yaw_belt", #garbage data
#                                    "kurtosis_yaw_belt") #garbage data
#                   )
# result4 <- generateModel(modelRds = modelFile
#                        , training = data4$training
#                        , testing = data4$testing )
#                        
# result4$confusionMatrix
# predict(result4$model, dataTestingPml)