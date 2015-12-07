# Kaggle Telstra competition
# https://www.kaggle.com/c/telstra-recruiting-network
# small dataset, multiclass, missing values, probably not many participants

source("util/funcs.R")
library(data.table)
library(ggplot2)
library(lattice)
library(dplyr)
library(tidyr)
library(pROC)
library(caret)
library(e1071)
library(glmnet)
library(RANN)
library(ipred)
# library(partykit)
# library(C50)
# library(ada)
library(gbm)

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
# 0.5869452 0.64486     same but only 5% val_indices_ori instead of 20% before (alpha = 0.55, lambda = 0.000447)
# 0.5818302 (TBD)       same with 3 seperate models - training params are different for the models
# 0.5686767 (TBD)       same with 20% val_indices_ori instead of 5%
# 0.5042326 (TBD)       with SB threshold 0 and NA imputation (for the test set) - probably overfitting
# 0.6023409 (TBD)       now excluding the val_indices_ori set from the binning - should report more accurate score
# 0.6033233 -           same but not tuning (surprisingly small effect)
# 0.6564631 - (??)      with 5% val_indices_ori instead of 20
# 0.6553885 0.66519     with 5% val_indices_ori and parameter tuning ... 
# 0.650963  0.65787     just a different random seed
# 0.6402995 0.66351     tune length 10 (instead of 3)
# 0.6410911 0.66377     no parameter tuning (keeping NA imputation and 5% val_indices_ori)
# 0.6973874 -           param tuning with C5.0 classifier
# 0.4177428 0.61144*    including sd and a few more and running GBM classifier - insanse difference val-LB!
# 0.6355389 0.67103     reviewed code, added graphs - this was with glmnet
# 0.6028989 0.62564     long GBM run but also lower correlation threshold (fewer vars)
# 0.5916207 0.62367     finetuned search grid (10% validation) 10/5 cv
# 0.6230756 0.61083*    but same position on LB :( more tuning and 1% validation (irrelevant - validate to LB)
#                       class 0 Fitting n.trees = 400, interaction.depth = 7, shrinkage = 0.05, n.minobsinnode = 10 on full training set
#                       class 1 Fitting n.trees = 400, interaction.depth = 6, shrinkage = 0.05, n.minobsinnode = 10 on full training set
#                       class 2 Fitting n.trees = 350, interaction.depth = 7, shrinkage = 0.05, n.minobsinnode = 10 on full training set
# 0.5926326 0.61089     Fitting n.trees = 600, interaction.depth = 9, shrinkage = 0.02, n.minobsinnode = 10 on full training set
# 0.6203718 0.61277     Fitting n.trees = 700, interaction.depth = 10, shrinkage = 0.02, n.minobsinnode = 10 on full training set
# 0.6096836 0.60836*    Correlation threshold .99 (instead of .95)
#                       Fitting n.trees = 650-700, interaction.depth = 10, shrinkage = 0.02, n.minobsinnode = 10 on full training set
#                       Took very long
#
# .. for the fun of it - try returning bin index instead of recoded values
# .. w/o tuning just use parameters

set.seed(491)
val_pct <- 1

trn_ori <- fread("data/train.csv")
tst_ori <- fread("data/test.csv")
log_feature <- fread("data/log_feature.csv")
event_type <- fread("data/event_type.csv")
resource_type <- fread("data/resource_type.csv")
severity_type <- fread("data/severity_type.csv")

# Original dimensions

train_ori_id <- trn_ori$id
train_ori_fault_severity <- trn_ori$fault_severity
test_ori_id <- tst_ori$id
val_indices_ori <- sample(nrow(trn_ori), val_pct*0.01*nrow(trn_ori))

# Join all together - this will result in a much taller dataset

train_joined <- left_join(trn_ori, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

test_joined <- left_join(tst_ori, resource_type) %>% 
  left_join(log_feature) %>%
  left_join(severity_type) %>%
  left_join(event_type)

val_indices_joined <- which( train_joined$id %in% train_ori_id[ val_indices_ori ])

# Check how many of the values in the test set are in the train set
# these are currently set to the average outcome
overlap <- sapply(names(test_joined), function(c) 
{return(paste(round(100*length(intersect(train_joined[[c]], test_joined[[c]]))/
                      length(unique(test_joined[[c]])),2),"%",sep=""))})
print("Overlap test-train set:")
print(overlap)

y <- data.frame(y0 = train_joined$fault_severity == 0,
                y1 = train_joined$fault_severity == 1,
                y2 = train_joined$fault_severity == 2)

train_joined <- select(train_joined, -fault_severity)

# predict 'fault_severity' (which is 0, 1 or 2)
colNames <- setdiff(names(train_joined), c("id"))
for (outcomeName in names(y)) {
  outcome <- y[[outcomeName]][-val_indices_joined]
  for (colName in colNames) {
    print(colName)
    
    binner <- createSymbin(train_joined[[colName]][-val_indices_joined], outcome, 0)
    
    binnedColName <- paste(colName,outcomeName,sep="_")
    train_joined[[binnedColName]] <- applySymbin(binner, train_joined[[colName]]) 
    test_joined[[binnedColName]] <- applySymbin(binner, test_joined[[colName]]) 
  }
}

myAUC <- function(response, predictor)
{
  df <- data.frame(predictor = predictor, response = response)
  df <- df[complete.cases(df),]
  return (auc(df$response, df$predictor))
}

# Only keep numerics
train_joined <- train_joined[, sapply(train_joined, is.numeric), with=F]
test_joined <- test_joined[, sapply(test_joined, is.numeric), with=F]

# Report on univariate AUC
univariates <- data.frame(
  y0 = sapply(names(train_joined), function(c) { return(myAUC(y$y0, train_joined[[c]]))}),
  y1 = sapply(names(train_joined), function(c) { return(myAUC(y$y1, train_joined[[c]]))}),
  y2 = sapply(names(train_joined), function(c) { return(myAUC(y$y2, train_joined[[c]]))}))
univariates$predictor <- row.names(univariates)
univariates <- gather( univariates, key="severity", value="auc", -predictor)
levels(univariates$severity) <- c(0,1,2)
print(ggplot(univariates, aes(x=predictor, y=auc, fill=severity)) + 
  geom_bar(stat="identity")+coord_flip()+geom_hline(yintercept=1.5))

# With a low threshold for SB, there can be NAs so we do imputation.
print("NA imputation on expanded data:")
cat("Any incomplete cases in development set?", any(!complete.cases(train_joined[-val_indices_joined])),fill=T)
cat("Any incomplete cases in validation set?",any(!complete.cases(train_joined[val_indices_joined,])),fill=T)
cat("Any incomplete cases in test set?",any(!complete.cases(test_joined)),fill=T)
fixUp <- preProcess(train_joined[-val_indices_joined,], method=c("bagImpute"), verbose=T) # medianImpute is faster but assumes independence
train_joined <- predict(fixUp, train_joined)
test_joined <- predict(fixUp, test_joined)
cat("Any incomplete cases in development set?", any(!complete.cases(train_joined[-val_indices_joined])),fill=T)
cat("Any incomplete cases in validation set?",any(!complete.cases(train_joined[val_indices_joined,])),fill=T)
cat("Any incomplete cases in test set?",any(!complete.cases(test_joined)),fill=T)

# Aggregate all numerics
print("Create aggregations:")
aggregate <- function(idCol, valCol, colNamePrefix) {
  ds <- data.frame(id = idCol, val = valCol)
  s <- group_by(ds, id) %>% dplyr::summarise( min = min(val),
                                              max = max(val),
                                              mean = mean(val),
#                                               first = first(val),
#                                               last = last(val),
                                              # seems to have a negative impact:
                                              mad = mad(val),
                                              #                                        distinct = n_distinct(val), # seems to have a negative impact
                                              # this introduces NAs:
                                              sd = sd(val), 
                                              median = median(val),
                                              iqr = IQR(val),
                                              n = n())   
  setnames(s, c("id",paste(colNamePrefix,names(s),sep="_")[2:length(names(s))]))
  return(s)
}

trainData <- NULL
testData <- NULL
for (numColName in setdiff(names(train_joined)[sapply(train_joined, is.numeric)], c("id"))) {
  print (numColName)
  trainAggregate <- aggregate( train_joined[["id"]], train_joined[[numColName]], numColName )
  testAggregate <- aggregate( test_joined[["id"]], test_joined[[numColName]], numColName )
  if (is.null(trainData)) {
    trainData <- trainAggregate
    testData <- testAggregate
  } else {
    trainData <- cbind(trainData, select(trainAggregate, -id))
    testData <- cbind(testData, select(testAggregate, -id))
  }
}

# return to the original order of the IDs
trainData <- trainData[ match(train_ori_id, trainData$id), ]
testData <- testData[ match(test_ori_id, testData$id), ]

# Some aggregators (like sd) can cause NAs
print("NA imputation on aggregated data:")
cat("Any incomplete cases in development set?", any(!complete.cases(trainData[-val_indices_joined])),fill=T)
cat("Any incomplete cases in val_indices_ori set?",any(!complete.cases(trainData[val_indices_joined,])),fill=T)
cat("Any incomplete cases in testData set?",any(!complete.cases(testData)),fill=T)
# NB if we want to be more efficicient we can do this for only the columns containing NA in the
# train set. The previous impute could check for columns in the test set.
# naCols <- which(sapply(trainData, function(x) {return(any(is.na(x)))}))
fixUp <- preProcess(trainData[-val_indices_ori,], method=c("bagImpute"), verbose=T) # medianImpute is faster but assumes independence
trainData <- predict(fixUp, trainData)
testData <- predict(fixUp, testData)
cat("Any incomplete cases in development set?", any(!complete.cases(trainData[-val_indices_joined])),fill=T)
cat("Any incomplete cases in val_indices_ori set?",any(!complete.cases(trainData[val_indices_joined,])),fill=T)
cat("Any incomplete cases in testData set?",any(!complete.cases(testData)),fill=T)

cat("Before zero variance:",dim(trainData),fill=T)

# (near) zero variance
nzv <- colnames(trainData)[nearZeroVar(trainData)]
cat("Removed near-zero variables:", length(nzv), 
    "(of", length(names(trainData)), "):", nzv, fill=T)
trainData <- trainData[,!(names(trainData) %in% nzv)]
testData  <- testData[,!(names(testData) %in% nzv)]

cat("Before linear combos:",dim(trainData),fill=T)

# drop the linear combinations
linearCombos <- colnames(trainData)[findLinearCombos(trainData)$remove]
cat("Removed linear combinations:", length(linearCombos), 
    "(of", length(names(trainData)), "):", linearCombos, fill=T)
trainData <- trainData[,!(names(trainData) %in% linearCombos)]
testData  <- testData[,!(names(testData) %in% linearCombos)]

cat("Before highly correlated:",dim(trainData),fill=T)

# remove highly correlated variables
trainCor <- cor( trainData, method='spearman')
correlatedVars <- colnames(trainData)[findCorrelation(trainCor, cutoff = 0.99, verbose = F)]
cat("Removed highly correlated cols:", length(correlatedVars), 
    "(of", length(names(trainData)), ")", correlatedVars, fill=T)
trainData <- trainData[,!(names(trainData) %in% correlatedVars)]
testData  <- testData[,!(names(testData) %in% correlatedVars)]

cat("Final set:",dim(trainData),fill=T)

# Report on univariate AUC
univariates2 <- data.frame(
  y0 = sapply(names(trainData), function(c) { return(myAUC(train_ori_fault_severity == 0, trainData[[c]]))}),
  y1 = sapply(names(trainData), function(c) { return(myAUC(train_ori_fault_severity == 1, trainData[[c]]))}),
  y2 = sapply(names(trainData), function(c) { return(myAUC(train_ori_fault_severity == 2, trainData[[c]]))}))
univariates2$predictor <- row.names(univariates2)
univariates2 <- gather( univariates2, key="severity", value="auc", -predictor)
levels(univariates2$severity) <- c(0,1,2)
univariates2$feature <- sub("(.*)_(y[012])_(.*)", "\\1", univariates2$predictor)
univariates2$aggregator <- sub("(.*)_(.*)","\\2",univariates2$predictor)
univariates2$target <- sub("(.*)_(y[012])_(.*)", "\\2", univariates2$predictor)
strEndsWith <- function(haystack, needle)
{
  hl <- nchar(haystack)
  nl <- nchar(needle)
  return(ifelse(nl>hl,F,substr(haystack, hl-nl+1, hl) == needle))
}
chopAggregator <- function(haystack, needle) 
{
  hl <- nchar(haystack)
  nl <- nchar(needle)
  return(ifelse(strEndsWith(haystack, needle),substr(haystack,1,hl-nl),haystack))
}
univariates2$target <- ifelse(strEndsWith(univariates2$feature, paste("_",univariates2$aggregator,sep="")),"",univariates2$target)
univariates2$feature <- chopAggregator(univariates2$feature, paste("_",univariates2$aggregator,sep=""))

# what do our aggregators do?
print(ggplot(group_by(univariates2, aggregator, severity) %>% dplyr::summarise(mean_auc = mean(auc)), 
             aes(x=aggregator, y=mean_auc, fill=severity)) + 
        geom_bar(stat="identity")+coord_flip()+geom_hline(yintercept=1.5))
# what do our features do?
print(ggplot(group_by(univariates2, feature, severity) %>% dplyr::summarise(mean_auc = mean(auc)), 
             aes(x=feature, y=mean_auc, fill=severity)) + 
        geom_bar(stat="identity")+coord_flip()+geom_hline(yintercept=1.5))
# what do our targets do?
print(ggplot(group_by(univariates2, target, severity) %>% dplyr::summarise(mean_auc = mean(auc)), 
             aes(x=target, y=mean_auc, fill=severity)) + 
        geom_bar(stat="identity")+coord_flip()+geom_hline(yintercept=1.5))
# all final predictors
print(ggplot(univariates2, aes(x=predictor, y=auc, fill=severity)) + 
        geom_bar(stat="identity")+coord_flip()+geom_hline(yintercept=1.5))

# descrCor2 <- cor(trainData)
# cat("Remaining correlations: ", summary(descrCor2[upper.tri(descrCor2)]), fill=T)

# see https://www.r-project.org/nosvn/conferences/useR-2013/Tutorials/kuhn/user_caret_2up.pdf
crossValidation <- trainControl(method = "repeatedcv", 
                                repeats = 5, number=10,
                                summaryFunction = mnLogLoss,
                                classProbs = TRUE,
                                verbose=T)

gbmGrid <-  expand.grid(interaction.depth = seq(8,10,by=1),  # more better at lower shrinkage
                        n.trees = seq(500,700,by=50), # more better at lower shrinkage
                        shrinkage = c(0.02,0.01,0.005), # 0.1 was worse
                        n.minobsinnode = 10) # doesnt seem to matter much

# now create model for all 3 outcomes seperately and score test set
trainResults <- data.frame(id = train_ori_id, 
                           obs = factor(paste("predict",train_ori_fault_severity,sep="_")))
testResults <- data.frame(id = test_ori_id)

# seperate two-class models:
for (target in c(0,1,2)) {
  cat("Predicting (single-class)", target, fill=T)
  trainData$y <- factor(ifelse(train_ori_fault_severity == target, "Yes", "No")) 
  
  model <- train(y ~ ., 
                 data = trainData[-val_indices_ori,], 
                 method = "gbm" # "glmnet" #"rpart"
                 #                    ,tuneLength = 10
                 ,tuneGrid = gbmGrid
                 ,metric = "logLoss"
                 ,maximize=F
                 ,trControl = crossValidation
                 ,verbose=F
  )
  
  trellis.par.set(caretTheme())
  print(plot(model))
  
  trainResults[[paste("predict",target,sep="_")]] <- predict.train(model, trainData, type="prob")[,2]
  testResults[[paste("predict",target,sep="_")]] <- predict.train(model, testData, type="prob")[,2]
}

# Get val_indices_ori score
score <- mnLogLoss(trainResults[val_indices_ori,], 
                   lev=levels(trainResults$obs)) # Caret multi-class LogLoss
cat("Validation score:", score, fill=T)

# Write submission score on test set
write.table(testResults, "submission.csv", 
            row.names = FALSE, quote = FALSE, sep = ",")

