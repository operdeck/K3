source("util/funcs.R")
library(data.table)
library(dplyr)
library(caret)
library(e1071)

# https://www.kaggle.com/c/telstra-recruiting-network/data

# Purely mechanical appraoch
# 
# Local     Public LB   Notes
# 0.6139549 0.67046     rpart - adding first/last/n aggregates did not help a bit, adding some others gives NA problems
# 0.475872  0.67804     using caret model tuning for rpart with ROC - perhaps the wrong metric

# TODO use caret for meta tuning? even for rpart? not sure what it optimizes right now - AUC?
# see https://www.r-project.org/nosvn/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf
# cvCtrl <- trainControl(method = "repeatedcv", repeats = 3,
# summaryFunction = twoClassSummary,
# classProbs = TRUE)
# set.seed(1)
# rpartTune <- train(Class ~ ., data = training, method = "rpart",
#                    tuneLength = 30,
#                    metric = "ROC",
#                    trControl = cvCtrl)

set.seed(1966)

train <- fread("data/train.csv")
test <- fread("data/test.csv")
log_feature <- fread("data/log_feature.csv")
event_type <- fread("data/event_type.csv")
resource_type <- fread("data/resource_type.csv")
severity_type <- fread("data/severity_type.csv")

train_id <- train$id
train_fault_severity <- train$fault_severity
test_id <- test$id
validation <- sample(1:nrow(train), 0.20*nrow(train))

# join all together - this will result in a much longer dataset

train <- left_join(train, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

test <- left_join(test, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

y <- data.frame(y0 = train$fault_severity == 0,
                y1 = train$fault_severity == 1,
                y2 = train$fault_severity == 2)

train <- select(train, -fault_severity)


# there are test locations not present in the train set but many are
# id's do not overlap

length(unique(test$location))
length( intersect(unique(train$location), unique(test$location)) )

# predict 'fault_severity' (which is 0, 1 or 2)

binCols <- setdiff(names(train), c("id"))
for (outcomeName in names(y)) {
  print(outcomeName)
  outcome <- y[[outcomeName]] # should this be excluding the validation set? [-validation]
  for (colName in binCols) {
    print(colName)
    
    binner <- createSymbin(train[[colName]], outcome) # should this be excluding the validation set?

    binnedColName <- paste(colName,outcomeName,sep="_")
    train[[binnedColName]] <- applySymbin(binner, train[[colName]]) 
    test[[binnedColName]] <- applySymbin(binner, test[[colName]]) 
  }
}

# aggregate all numerics

aggregate <- function(idCol, valCol, colNamePrefix) {
  ds <- data.frame(id = idCol, val = valCol)
  s <- group_by(ds, id) %>% summarise( min = min(val),
                                       max = max(val),
                                       mean = mean(val),
                                       first = first(val),
                                       last = last(val),
#                                        distinct = n_distinct(val),
#                                        sum = sum(val),
#                                        sd = sd(val, na.rm=T),
#                                        median = median(val),
#                                        iqr = IQR(val),
                                       n = n())   
  setnames(s, c("id",paste(colNamePrefix,names(s),sep="_")[2:length(names(s))]))
  return(s)
}

trainModelSet <- NULL
testModelSet <- NULL
for (numColName in setdiff(names(train)[sapply(train, is.numeric)], c("id"))) {
  print (numColName)
  trainAggregate <- aggregate( train[["id"]], train[[numColName]], numColName )
  testAggregate <- aggregate( test[["id"]], test[[numColName]], numColName )
  if (is.null(trainModelSet)) {
    trainModelSet <- trainAggregate
    testModelSet <- testAggregate
  } else {
    trainModelSet <- cbind(trainModelSet, select(trainAggregate, -id))
    testModelSet <- cbind(testModelSet, select(testAggregate, -id))
  }
}

# return to the original order of the IDs
trainModelSet <- trainModelSet[ match(train_id, trainModelSet$id), ]
testModelSet <- testModelSet[ match(test_id, testModelSet$id), ]

# drop the linear combinations
linearCombos <- colnames(trainModelSet)[findLinearCombos(trainModelSet)$remove]
cat("Removed linear combinations:", length(linearCombos), 
    "(of", length(names(trainModelSet)), "):", linearCombos, fill=T)
trainModelSet <- trainModelSet[,!(names(trainModelSet) %in% linearCombos)]
testModelSet  <- testModelSet[,!(names(testModelSet) %in% linearCombos)]

# remove highly correlated variables
trainCor <- cor( trainModelSet, method='spearman')
correlatedVars <- colnames(trainModelSet)[findCorrelation(trainCor, cutoff = 0.99, verbose = F)]
cat("Removed highly correlated cols:", length(correlatedVars), 
    "(of", length(names(trainModelSet)), ")", correlatedVars, fill=T)
trainModelSet <- trainModelSet[,!(names(trainModelSet) %in% correlatedVars)]
testModelSet  <- testModelSet[,!(names(testModelSet) %in% correlatedVars)]

# Kaggle's LogLoss evaluation function
ll <- function(act, pred)
{
  eps <- 1e-15
  if (!is.logical(act)) {
    stop("Logloss expects a logical as first argument")
  }
  ll <- -1*sum(act*log(pmax(pmin(pred,1-eps),eps))) / length(act)
  return(ll)
}

# see https://www.r-project.org/nosvn/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf
cvCtrl <- trainControl(method = "repeatedcv", repeats = 3, number=10,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE)

# now create LR model for all 3 outcomes seperately and score test set
trainResults <- data.frame(id = train_id, y = train_fault_severity)
testResults <- data.frame(id = test_id)
for (target in c(0,1,2)) {
  cat("Predicting ", target, fill=T)
  trainModelSet$y <- factor(ifelse(train_fault_severity == target, "Yes", "No")) 
  
  model <- train(y ~ ., data = trainModelSet[-validation], method = "rpart"
#                  ,tuneLength = 30
#                  ,metric = "ROC"
#                  ,trControl = cvCtrl
                 )

  # probs <- predict.train(model, trainModelSet)#, type="prob")
  # confusionMatrix(probs, trainModelSet$y)
  trainResults[[paste("predict",target,sep="_")]] <- predict.train(model, trainModelSet, type="prob")[,2]
  testResults[[paste("predict",target,sep="_")]] <- predict.train(model, testModelSet, type="prob")[,2]
}

# evaluate...
valSet <- trainResults[validation,]

score <- sum(ll(valSet$y==0, valSet$predict_0),
             ll(valSet$y==1, valSet$predict_1),
             ll(valSet$y==2, valSet$predict_2))

print(score)

write.table(testResults, "submission.csv", row.names = FALSE, quote = FALSE, sep = ",")



