# Kaggle Telstra competition
# https://www.kaggle.com/c/telstra-recruiting-network
# small dataset, multiclass, missing values, probably not many participants

source("util/funcs.R")
library(data.table)
library(ggplot2)
library(lattice)
library(plyr) # used by caret
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
# 0.4349778 0.58856*    With feature_log aggregates and clustering. 20% val. 95% corr th. Removed some slow aggregates. No GBM tuning.
# 0.3883476 0.59890     1% val, few aggregates re-enabled (making it much slower)
# 0.3877019 0.59891     1% val, most extra aggregates gone
# 0.3744574 0.60208     1% val as opposed to 20% earlier - what the hack - maybe overfitting is the bigger problem
# 0.4783086 0.60074     40% validation... Final set: 7381 x 268
# 0.436575 settings back to (*) I hope - score is similar but not the same, only diff seems to a name (n vs sum)
# 0.4381896 0.60883     Adding cluster index for all tables, but not joining in all aggregates (so far fewer predictors)
# 0.4272738 0.60723     at 1% validation
# 0.4413344 0.62409     clusters as symbolic, rank instead of recode
# 0.444007  0.60682     same, recode instead of rank
# 0.4390447 0.59842     few more aggregates (median etc)
# 0.4369168 (TODO)      packages updates and back to bagImputation (vs median); 10% val Final set: 7381 x 70
# 0.4440401 (TODO)      same with xgbTree
# Fitting nrounds = 4000, eta = 0.01, max_depth = 10, gamma = 1, colsample_bytree = 1, min_child_weight = 6 on full training set
# 0.4393507 0.67235     grmpf - strange rank index added
# 0.5326185 0.55223*    xgb with param tuning and no supervised binning

# .. maybe add another cluster as well on event_type?? and do that first??
# .. for the fun of it - try returning bin index instead of recoded values
# .. w/o tuning just use parameters
# time for XGB?


set.seed(491)
val_fraction <- 0.10
doMultiClass <- T # multi-class or seperate predictions for each level

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
val_indices_ori <- sample(nrow(trn_ori), val_fraction*nrow(trn_ori))

# Just for exploration. This shows that
#   incident [ 1:1 ] severity_type (5)
#   incident [ 1:N ] log_feature (386), event_type (53), resource_type (10)
#
# Test set: 11171 x 2; 1039 unique locations (842 overlapping with train set)
# Train set: 7381 x 2; 929 unique locations

relations <- data.frame(
  size = c(nrow(trn_ori), nrow(tst_ori)),
  log_feature_join = c(nrow(left_join(trn_ori, log_feature)), nrow(left_join(tst_ori, log_feature))),
  overlap = c(length(intersect( trn_ori$id, log_feature$id )), length(intersect( tst_ori$id, log_feature$id ))),
  resource_type_join = c(nrow(left_join(trn_ori, resource_type)), nrow(left_join(tst_ori, resource_type))),
  overlap = c(length(intersect( trn_ori$id, resource_type$id )), length(intersect( tst_ori$id, resource_type$id ))),
  severity_type_join = c(nrow(left_join(trn_ori, severity_type)), nrow(left_join(tst_ori, severity_type))),
  overlap = c(length(intersect( trn_ori$id, severity_type$id )), length(intersect( tst_ori$id, severity_type$id ))),
  event_type_join = c(nrow(left_join(trn_ori, event_type)), nrow(left_join(tst_ori, event_type))),
  overlap = c(length(intersect( trn_ori$id, event_type$id )), length(intersect( tst_ori$id, event_type$id ))))
rownames(relations) <- c('train','test')
print(relations)

# Hot-one encoding of "severity_type"
hotOneSeverityType <- data.table(id=severity_type$id,
                                 model.matrix(~.-1,
                                              mutate(severity_type, sev_type_=sub(".* ([0-9]+)", "\\1", severity_type)) %>% 
                                                select(sev_type_)))

# Hot-one encoding of "resource_type"
hotOneResourceType <- data.table(id=resource_type$id,
                                 model.matrix(~.-1,
                                              mutate(resource_type, res_type_=sub(".* ([0-9]+)", "\\1", resource_type)) %>% 
                                                select(res_type_)))

# Join all together - this will result in a much taller dataset
train_joined <- trn_ori %>%
  left_join(log_feature) %>%
  left_join(event_type) %>%
  left_join(hotOneSeverityType) %>% 
  left_join(hotOneResourceType)
test_joined <- tst_ori %>%
  left_join(log_feature) %>%
  left_join(event_type) %>%
  left_join(hotOneSeverityType) %>% 
  left_join(hotOneResourceType)
val_indices_joined <- which( train_joined$id %in% train_ori_id[ val_indices_ori ])

# print(filter(train_joined, location=="location 1111"))

# Try un-supervised clustering for "location", "log_feature" and "event_type"
clusterSource <- rbind(select(trn_ori, -fault_severity), tst_ori) %>%
        left_join(log_feature) %>%
        left_join(event_type) %>%
        left_join(hotOneSeverityType) %>% 
        left_join(hotOneResourceType) %>%
  select(-id)

# Create clusters
symCols <- which(sapply(clusterSource, class) == "character")
for (clusterColIndex in symCols) {
  clusterColName <- names(clusterSource)[clusterColIndex]
  
  # Data table with the cluster key plus all numeric columns
  clusterSourceNums <- data.table(clusterCol=clusterSource[[clusterColIndex]])
  clusterSourceNums <- cbind(clusterSourceNums, clusterSource[,-symCols,with=F])
  
  # Summary of the table keyed by cluster key and aggregations of all numeric columns
  summarized <- clusterSourceNums %>% group_by(clusterCol) %>% dplyr::summarise(cnt_clusterCol = n())
  for (c in 2:ncol(clusterSourceNums)) {
    xx <- data.frame(clusterCol=clusterSourceNums$clusterCol, val=clusterSourceNums[[c]]) %>% group_by(clusterCol) %>% 
      dplyr::summarise(x1 = sum(val))
    names(xx) <- c('clusterCol', paste(names(clusterSourceNums)[c],"sum",sep="_"))
    summarized <- cbind(summarized, subset(xx,select=2))
  }
  
  # Cluster the summarized table
  sc <- scale(select(summarized, -clusterCol))
  wss <- (nrow(sc)-1)*sum(apply(sc,2,var))
  maxClusters <- min(100, nrow(sc)/2)
  for (i in 2:maxClusters) wss[i] <- sum(kmeans(sc, centers=i)$withinss)
  print(qplot(x=1:maxClusters,y=wss,geom="line")+
          xlab("Number of Clusters")+
          ylab("Within groups sum of squares")+
          ggtitle(paste("Cluster analysis for", clusterColName)))
  
  nClusters <- NA
  if (clusterColName == "event_type") {
    nClusters <- 5
  } else if (clusterColName == "log_feature") {
    nClusters <- 10
  } else if (clusterColName == "location") {
    nClusters <- 20
  }
  # Create a table with just the cluster column and the cluster index and join
  # these to the train/test sets. Potentially we could also add all summarized/aggregated
  # other fields (cbind with summarized) but I dont think that will add much.
  clusterIndex <- data.table( clusterCol = summarized$clusterCol,
                              clusterFit = as.character(kmeans(sc, nClusters)$cluster ))
  names(clusterIndex) <- c(clusterColName, paste(clusterColName, "cluster", sep="_"))
  train_joined <- left_join(train_joined, clusterIndex)
  test_joined <- left_join(test_joined, clusterIndex)
}

# Straightforward encoding or Supervised binning for "location", "log_feature" and "event_type" and all other
# remaining symbolics
# (they have too many values for hot-one encoding)
y <- data.frame(y0 = train_joined$fault_severity == 0,
                y1 = train_joined$fault_severity == 1,
                y2 = train_joined$fault_severity == 2)
train_joined <- select(train_joined, -fault_severity)

doSymBinning <- F

colNames <- names(train_joined)[which(sapply(train_joined, class) == "character")]
for (colName in colNames) {
  if (!doSymBinning) {
    binnedColName <- paste(colName,"factor",sep="_")
    lev <- levels(as.factor(c(train_joined[[colName]], 
                              test_joined[[colName]])))
    train_joined[[binnedColName]] <- as.integer(factor(train_joined[[colName]], levels=lev) )
    test_joined[[binnedColName]] <- as.integer(factor(test_joined[[colName]], levels=lev) )
  } else {
    for (outcomeName in names(y)) {
      outcome <- y[[outcomeName]][-val_indices_joined]
      binnedColName <- paste(colName,outcomeName,sep="_")
      cat("Symbolic binning",colName,"->",binnedColName,fill=T)
      binner <- createSymbin(train_joined[[colName]][-val_indices_joined], outcome, 0)
      train_joined[[binnedColName]] <- applySymbin(binner, train_joined[[colName]]) 
      test_joined[[binnedColName]] <- applySymbin(binner, test_joined[[colName]]) 
    }
    # An extra variable that indicates the order _y0 > _y1 > _y2 etc?
    # rank y0 y1 Y2  3 x rank y0 + rank y1
    #       1  2  3  5
    #       1  3  2  6
    #       2  1  3  7
    #       2  3  1  9
    #       3  1  2  10
    #       3  2  1  12
  #   train_recodes <- data.frame(y0 = train_joined[[paste(colName,"y0",sep="_")]],
  #                               y1 = train_joined[[paste(colName,"y1",sep="_")]],
  #                               y2 = train_joined[[paste(colName,"y2",sep="_")]])
  #   train_recodes <- cbind(train_recodes, t(apply(train_recodes, 1, rank, ties.method= "first")))
  #   train_joined[[paste(colName, "binorder", sep="_")]] <- train_recodes[,4]*3+train_recodes[,5]
  # 
  #   test_recodes <- data.frame(y0 = test_joined[[paste(colName,"y0",sep="_")]],
  #                               y1 = test_joined[[paste(colName,"y1",sep="_")]],
  #                               y2 = test_joined[[paste(colName,"y2",sep="_")]])
  #   test_recodes <- cbind(test_recodes, t(apply(test_recodes, 1, rank, ties.method= "first")))
  #   test_joined[[paste(colName, "binorder", sep="_")]] <- test_recodes[,4]*3+test_recodes[,5]
  }
}

# Check how many of the values in the test set are in the train set
# these are currently set to the average outcome
overlap <- sapply(names(test_joined), function(c) 
{return(paste(round(100*length(intersect(train_joined[[c]], test_joined[[c]]))/
                      length(unique(test_joined[[c]])),2),"%",sep=""))})
print("How many in the test set are contained in the train set:")
print(overlap)

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
print(head(arrange(univariates, desc(auc))),10)

# With a low threshold for SB, there can be NAs so we do imputation.
print("NA imputation on expanded data:")
cat("Any incomplete cases in development set?", any(!complete.cases(train_joined[-val_indices_joined])),fill=T)
cat("Any incomplete cases in validation set?",any(!complete.cases(train_joined[val_indices_joined,])),fill=T)
cat("Any incomplete cases in test set?",any(!complete.cases(test_joined)),fill=T)
fixUp <- preProcess(train_joined[-val_indices_joined,], method=c("bagImpute"), verbose=T) # bagImpute / medianImpute is faster but assumes independence
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
#                                               mad = mad(val),
                                              #                                        distinct = n_distinct(val), # seems to have a negative impact
                                              # this introduces NAs:
                                              sd = sd(val), 
#                                               median = median(val),
#                                               iqr = IQR(val),
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
cat("Any incomplete cases in validation set?",any(!complete.cases(trainData[val_indices_joined,])),fill=T)
cat("Any incomplete cases in testData set?",any(!complete.cases(testData)),fill=T)
fixUp <- preProcess(trainData[-val_indices_ori,], method=c("bagImpute"), verbose=T) # bagImpute/medianImpute is faster but assumes independence
trainData <- predict(fixUp, trainData)
testData <- predict(fixUp, testData)
cat("Any incomplete cases in development set?", any(!complete.cases(trainData[-val_indices_joined])),fill=T)
cat("Any incomplete cases in validation set?",any(!complete.cases(trainData[val_indices_joined,])),fill=T)
cat("Any incomplete cases in testData set?",any(!complete.cases(testData)),fill=T)

cat("Data set size before variable selection:",dim(trainData),fill=T)

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

print(head(arrange(univariates2, desc(auc))),10)

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
crossValidation <- trainControl(#method = "repeatedcv",
#                                 method = "none", # Fitting models without parameter tuning
                                method = "cv",
#                                 repeats = 5, 
                                number=5,
                                summaryFunction = mnLogLoss,
                                classProbs = TRUE,
                                verbose=T)

# gbmGrid <-  expand.grid(interaction.depth = seq(8,10,by=1),  # more better at lower shrinkage
#                         n.trees = seq(500,700,by=50), # more better at lower shrinkage
#                         shrinkage = c(0.02,0.01,0.005), # 0.1 was worse
#                         n.minobsinnode = 10) # doesnt seem to matter much
gbmGrid <- data.frame(interaction.depth = 10,
                      n.trees = 650,
                      shrinkage = 0.02,
                      n.minobsinnode = 10)

xgbGrid <- expand.grid(nrounds = 500,
                      eta = c(0.05, 0.1, 0.2), # 0.01
                      max_depth = seq(4,10,by=2),
                      gamma = c(1, 2, 4), 
                      colsample_bytree = 1, 
                      min_child_weight = c(4,6,10))

trainResults <- data.frame(id = train_ori_id, 
                           obs = factor(paste("predict",train_ori_fault_severity,sep="_")))
testResults <- data.frame(id = test_ori_id)

if (doMultiClass) {
  # single multi-class model
  cat("Predicting (multi-class)", fill=T)
  trainData$y <- as.factor(train_ori_fault_severity)
  levels(trainData$y) <- paste("predict",levels(trainData$y),sep="_")
  model <- train(y ~ ., 
                 data = trainData[-val_indices_ori,], 
                 method = "xgbTree" # gbm "glmnet" #"rpart"
                 #                    ,tuneLength = 10
                 ,tuneGrid = xgbGrid # gbmGrid
                 ,metric = "logLoss"
                 ,maximize=F
                 ,trControl = crossValidation
                 ,verbose=F
  )

  if (crossValidation$method != "none") {
    trellis.par.set(caretTheme())
    print(plot(model))
  }
  
  trainResults <- cbind(trainResults, predict.train(model, trainData, type="prob"))
  testResults <- cbind(testResults, predict.train(model, testData, type="prob"))
} else {
  # seperate two-class models for each outcome level:
  for (target in c(0,1,2)) {
    cat("Predicting (single-class)", target, fill=T)
    trainData$y <- factor(ifelse(train_ori_fault_severity == target, "Yes", "No")) 
    
    model <- train(y ~ ., 
                   data = trainData[-val_indices_ori,], 
                   method = "xgbTree" # gbm "glmnet" #"rpart"
                   #                    ,tuneLength = 10
                   ,tuneGrid = xgbGrid # gbmGrid
                   ,metric = "logLoss"
                   ,maximize=F
                   ,trControl = crossValidation
                   ,verbose=F
    )
    
    if (crossValidation$method != "none") {
      trellis.par.set(caretTheme())
      print(plot(model))
    }
    
    trainResults[[paste("predict",target,sep="_")]] <- predict.train(model, trainData, type="prob")[,2]
    testResults[[paste("predict",target,sep="_")]] <- predict.train(model, testData, type="prob")[,2]
  }
}

# if validation not none; model$bestTune / model$finalModel
# Get val_indices_ori score

if (crossValidation$method != "none") {
  bestIdx <- as.integer(rownames(model$bestTune))
  bestScore <- model$results$logLoss[bestIdx]
} else {
  bestScore <- mnLogLoss(trainResults[val_indices_ori,], 
                     lev=levels(trainResults$obs)) # Caret multi-class LogLoss
}
cat("Validation score:", bestScore, fill=T)

# Write submission score on test set
write.table(testResults, "submission.csv", 
            row.names = FALSE, quote = FALSE, sep = ",")

