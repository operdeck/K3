# Kaggle Telstra competition
# https://www.kaggle.com/c/telstra-recruiting-network
# small dataset, multiclass, missing values, probably not many participants

source("util/funcs.R")
library(data.table)
library(dplyr)
library(caret)
library(e1071)
library(glmnet)
library(RANN)
library(ipred)
# library(partykit)
# library(C50)
library(ada)

# https://www.kaggle.com/c/telstra-recruiting-network/data

# Purely mechanical approach
# 
# Local     Public LB   Notes
# 0.6139549 0.67046*    rpart - adding first/last/n aggregates did not help a bit, adding some others gives NA problems
# 0.475872  0.67804     using caret model tuning for rpart with ROC - perhaps the wrong metric
# 0.6523214 0.71524     one multiclass model (rpart) instead of 3 seperate ones - so that's not a great idea
# 0.5693206 0.64395*    glmnet instead of rpart - no additional parameters
# 0.6144861 0.67153     rpart with two more aggregated variables (sum and n_distinct), reporting via Caret now
# 0.6167727 0.67118     rpart without these two aggregated vars but w/o 'first' and 'last'
# 0.6139549 -           ok back to same score, so 'first' and 'last' aggregates did matter
# 0.5687924 -           multi-class glmnet with 10/3 parameter tuning (alpha = 0.55, lambda = 0.000447)
# 0.5869452 0.64486     same but only 5% validation instead of 20% before (alpha = 0.55, lambda = 0.000447)
# 0.5818302 (TBD)       same with 3 seperate models - training params are different for the models
# 0.5686767 (TBD)       same with 20% validation instead of 5%
# 0.5042326 (TBD)       with SB threshold 0 and NA imputation (for the test set) - probably overfitting
# 0.6023409 (TBD)       now excluding the validation set from the binning - should report more accurate score
# 0.6033233 -           same but not tuning (surprisingly small effect)
# 0.6564631 - (??)      with 5% validation instead of 20
# 0.6553885 - (???)     with 5% validation and parameter tuning ... 
# 0.650963  -           just a different random seed
# 0.6411956 -           and another random seed ...
# 
# ..        ..          nice to try C50 or ada but they take forever
# TODO: submit for tuned glmnet, distinct models, NA imputation, proper validation scoring and 5% validation

set.seed(491)

trn_ori <- fread("data/train.csv")
tst_ori <- fread("data/test.csv")
log_feature <- fread("data/log_feature.csv")
event_type <- fread("data/event_type.csv")
resource_type <- fread("data/resource_type.csv")
severity_type <- fread("data/severity_type.csv")

train_id <- trn_ori$id
train_fault_severity <- trn_ori$fault_severity
test_id <- tst_ori$id
validation <- sample(1:nrow(trn_ori), 0.05*nrow(trn_ori))

# join all together - this will result in a much taller dataset

train <- left_join(trn_ori, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

test <- left_join(tst_ori, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

# Check how many of the values in the test set are in the train set
# these are currently set to the average outcome
overlap <- sapply(names(test), function(c) 
  {return(length(intersect(train[[c]], test[[c]]))/length(unique(test[[c]])))})
print("Overlap test-train set:")
print(overlap)

y <- data.frame(y0 = train$fault_severity == 0,
                y1 = train$fault_severity == 1,
                y2 = train$fault_severity == 2)

train <- select(train, -fault_severity)

# predict 'fault_severity' (which is 0, 1 or 2)
validationSetRows <- which( train$id %in% train_id[ validation ])
binCols <- setdiff(names(train), c("id"))
for (outcomeName in names(y)) {
  print(outcomeName)
  outcome <- y[[outcomeName]][-validationSetRows]
  for (colName in binCols) {
    print(colName)
    
    binner <- createSymbin(train[[colName]][-validationSetRows], outcome, 0)
    
    binnedColName <- paste(colName,outcomeName,sep="_")
    train[[binnedColName]] <- applySymbin(binner, train[[colName]]) 
    test[[binnedColName]] <- applySymbin(binner, test[[colName]]) 
  }
}

# With a low threshold for SB, there can be NAs so we do imputation. Also, we only continue with the numerics now.
print("NA imputation:")
train <- train[, sapply(train, is.numeric), with=F]
test <- test[, sapply(test, is.numeric), with=F]
cat("Any incomplete cases in development set?", any(!complete.cases(train[-validationSetRows])),fill=T)
cat("Any incomplete cases in validation set?",any(!complete.cases(train[validationSetRows,])),fill=T)
cat("Any incomplete cases in test set?",any(!complete.cases(test)),fill=T)
fixUp <- preProcess(train[-validationSetRows], method=c("bagImpute"), verbose=T) # medianImpute is faster but assumes independence
train <- predict(fixUp, train)
test <- predict(fixUp, test)
cat("Any incomplete cases in development set?", any(!complete.cases(train[-validationSetRows])),fill=T)
cat("Any incomplete cases in validation set?",any(!complete.cases(train[validationSetRows,])),fill=T)
cat("Any incomplete cases in test set?",any(!complete.cases(test)),fill=T)

# Aggregate all numerics
print("Create aggregations:")
aggregate <- function(idCol, valCol, colNamePrefix) {
  ds <- data.frame(id = idCol, val = valCol)
  s <- group_by(ds, id) %>% dplyr::summarise( min = min(val),
                                       max = max(val),
                                       mean = mean(val),
                                       first = first(val),
                                       last = last(val),
                                       # seems to have a negative impact:
#                                        mad = mad(val),
#                                        distinct = n_distinct(val), # seems to have a negative impact
#                                        sum = sum(val), 
                                       # this introduces NAs
#                                        sd = sd(val), 
#                                        median = median(val),
#                                        iqr = IQR(val),
                                       n = n())   
  setnames(s, c("id",paste(colNamePrefix,names(s),sep="_")[2:length(names(s))]))
  return(s)
}

trainData <- NULL
testData <- NULL
for (numColName in setdiff(names(train)[sapply(train, is.numeric)], c("id"))) {
  print (numColName)
  trainAggregate <- aggregate( train[["id"]], train[[numColName]], numColName )
  testAggregate <- aggregate( test[["id"]], test[[numColName]], numColName )
  if (is.null(trainData)) {
    trainData <- trainAggregate
    testData <- testAggregate
  } else {
    trainData <- cbind(trainData, select(trainAggregate, -id))
    testData <- cbind(testData, select(testAggregate, -id))
  }
}

# return to the original order of the IDs
trainData <- trainData[ match(train_id, trainData$id), ]
testData <- testData[ match(test_id, testData$id), ]

# (near) zero variance
nzv <- colnames(trainData)[nearZeroVar(trainData)]
cat("Removed near-zero variables:", length(nzv), 
    "(of", length(names(trainData)), "):", nzv, fill=T)
trainData <- trainData[,!(names(trainData) %in% nzv)]
testData  <- testData[,!(names(testData) %in% nzv)]

# drop the linear combinations
linearCombos <- colnames(trainData)[findLinearCombos(trainData)$remove]
cat("Removed linear combinations:", length(linearCombos), 
    "(of", length(names(trainData)), "):", linearCombos, fill=T)
trainData <- trainData[,!(names(trainData) %in% linearCombos)]
testData  <- testData[,!(names(testData) %in% linearCombos)]

# remove highly correlated variables
trainCor <- cor( trainData, method='spearman')
correlatedVars <- colnames(trainData)[findCorrelation(trainCor, cutoff = 0.99, verbose = F)]
cat("Removed highly correlated cols:", length(correlatedVars), 
    "(of", length(names(trainData)), ")", correlatedVars, fill=T)
trainData <- trainData[,!(names(trainData) %in% correlatedVars)]
testData  <- testData[,!(names(testData) %in% correlatedVars)]

# descrCor2 <- cor(trainData)
# cat("Remaining correlations: ", summary(descrCor2[upper.tri(descrCor2)]), fill=T)

# see https://www.r-project.org/nosvn/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf
cvCtrl <- trainControl(method = "repeatedcv", repeats = 3, number=10,
                       summaryFunction = mnLogLoss,
                       classProbs = TRUE,
                       verbose=T)

# now create LR model for all 3 outcomes seperately and score test set
trainResults <- data.frame(id = train_id, obs = factor(paste("predict",train_fault_severity,sep="_")))
testResults <- data.frame(id = test_id)

# seperate two-class models:
if (T) {
  for (target in c(0,1,2)) {
    cat("Predicting (single-class)", target, fill=T)
    trainData$y <- factor(ifelse(train_fault_severity == target, "Yes", "No")) 
    
    model <- train(y ~ ., data = trainData[-validation], method = "glmnet" # "glmnet" #"rpart"
                   ,tuneLength = 3
                   ,metric = "logLoss"
                   ,maximize=F
                   ,trControl = cvCtrl
    )
  
    trainResults[[paste("predict",target,sep="_")]] <- predict.train(model, trainData, type="prob")[,2]
    testResults[[paste("predict",target,sep="_")]] <- predict.train(model, testData, type="prob")[,2]
  }
} else {
  # single multi-class model
  print("Predicting (multi-class)")
  trainData$y <- factor(paste("predict",train_fault_severity,sep="_")) 
  model <- train(y ~ ., data = trainData[-validation], method = "glmnet" # "glmnet" #"rpart"
                 #                  ,tuneLength = 30
                 ,metric = "logLoss"
                 ,maximize=F
                 ,trControl = cvCtrl
  )
  trainResults <- cbind(trainResults, predict.train(model, trainData, type="prob"))
  testResults <- cbind(testResults, predict.train(model, testData, type="prob"))
}

# Get validation score
score <- mnLogLoss(trainResults[validation,], lev=levels(trainResults$obs)) # Caret multi-class LogLoss
cat("Validation score:", score, fill=T)

# Write submission score on test set
write.table(testResults, "submission.csv", row.names = FALSE, quote = FALSE, sep = ",")

