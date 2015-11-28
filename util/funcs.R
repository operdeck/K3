# load required packages
library(data.table)
require(bit64)
library(tidyr)
library(ggplot2) 
library(scales)
library(caret)
library(lubridate)
library(corrplot)
#library(plyr)
library(dplyr)
library(scales)
library(sm)
library(Hmisc)

# Metrix - https://github.com/benhamner/Metrics/blob/master/R/R/metrics.r

# Create a binning object for symbolic binning of a vector of values. Values
# with a frequency below a threshold will be represented by a 'residual' bin.
# There is a seperate bin for missing values.
#
# Example output:
# 
#        val cases avgoutcome      freq binIndex binRank
# 1        B     2        0.0 0.2857143        1       1
# 2        A     2        1.0 0.2857143        2       3
# 3 RESIDUAL     1        1.0 0.1428571        3       4
# 4  MISSING     2        0.5 0.2857143        4       2
# 
createSymbin <- function(val, outcome, threshold = 0.001) 
{
  if (!is.logical(outcome) && !is.integer(outcome)) {
    stop("expects a logical or integer as 2nd argument")
  }
  total <- length(outcome)
  
  # Create one bin for 'NA' values
  isNA <- is.na(val)
  result <- data.frame(c('MISSING'),sum(isNA),mean(outcome[isNA],na.rm=T))
  setnames(result, c('val', 'cases', 'avgoutcome'))
  result$freq <- result$cases/total
  
  if (!all(isNA)) {
    # Create bins for all other values
    df <- data.frame(val[!isNA], outcome[!isNA]) 
    setnames(df, c('val','outcome'))
    g <- group_by(df, val) %>% 
      dplyr::summarise(cases = n(), 
                       avgoutcome = mean(outcome,na.rm=T), 
                       freq = cases/total,
                       isAboveThreshold = freq>=threshold)

    if (any(g$isAboveThreshold)) {
      g$val <- as.character(g$val)
      # Create single bin for residual group and concatenate the bins
      result <- rbind(filter(g, isAboveThreshold) %>% select(-isAboveThreshold), 
                      c(NA,NA,NA,NA), result)
      residual_cases <- sum(g$cases[!g$isAboveThreshold])
      result$avgoutcome[nrow(result)-1] <- 
        sum(g$avgoutcome[!g$isAboveThreshold] * g$cases[!g$isAboveThreshold]) / residual_cases
    } else {
      # Special care if all are residual
      result <- rbind(result, result) # first one will be overwritten
      result$val <- as.character(result$val)
      residual_cases <- sum(g$cases)
      result$avgoutcome[nrow(result)-1] <- sum(g$avgoutcome*g$cases)/residual_cases
    }
  } else {
    # Special care if all values are missing
    result <- rbind(result, result) # first one will be overwritten
    result$val <- as.character(result$val)
    residual_cases <- 0
    result$avgoutcome[nrow(result)-1] <- NA
  } 
  result$val[nrow(result)-1] <- 'RESIDUAL'
  result$cases[nrow(result)-1] <- residual_cases
  result$freq[nrow(result)-1] <- residual_cases/total
  result$avgoutcome[is.na(result$avgoutcome)] <- NA # get rid of NaN, replace by NA
  
  result$binIndex <- 1:nrow(result)
  result$binRank <- rank(result$avgoutcome, ties.method="first")
  
  return(result)
}

# Apply sym binning to a vector of values, returning a vector of bin indices
# Values not in the binning will get the 'residual' bin index
applySymbin.internal <- function(binning, values) 
{
  if (nrow(binning) == 1) {
    return (rep(1, length(values)))  
  } else {
    df <- data.frame(as.character(values), stringsAsFactors = F)
    setnames(df, c('val'))
    r <- left_join(df, binning[seq(1:(nrow(binning)-2)),], by="val")$binIndex
    
    r[is.na(values)] <- nrow(binning)
    # if there's still NAs, these are because the values are residuals
    r[is.na(r)] <- nrow(binning)-1
    
    return(r)
  }
}

# Apply sym binning to a vector of values, returning a vector of recoded values
# Values not in the binning will get the 'residual' recoding
applySymbin <- function(binning, values) 
{
  return (binning$avgoutcome[applySymbin.internal(binning, values)])
}

# Apply sym binning to a vector of values, returning a vector of ranks of the outcomes
# Values not in the binning will get the rank of the 'residual' bin
applySymbinRank <- function(binning, values) 
{
  return (binning$binRank[applySymbin.internal(binning, values)])
}

# Plot symbolic data analysis for one set of vectors
sb.plotOne <- function(binning, 
                       ds_dev, ds_val, ds_tst,
                       fieldName, outcomeName,
                       plotFolder = NULL)
{
  ds_dev_bins <- applySymbin.internal(binning, ds_dev[[fieldName]])
  df_dev <- data.frame( ds_dev_bins, ds_dev[,outcomeName])
  names(df_dev) <- c('binRank','beh')
  
  ds_val_bins <- applySymbin.internal(binning, ds_val[[fieldName]])
  df_val <- data.frame( ds_val_bins, ds_val[,outcomeName])
  names(df_val) <- c('binRank','beh')
  
  ds_tst_bins <- applySymbin.internal(binning, ds_tst[[fieldName]])
  df_tst <- data.frame( ds_tst_bins )
  names(df_tst) <- c('binRank')
  
  rs_dev <- group_by(df_dev, binRank) %>% dplyr::summarise( dev_f=n()/nrow(df_dev), avgoutcome=mean(beh,na.rm=T) )
  rs_val <- group_by(df_val, binRank) %>% dplyr::summarise( val_f=n()/nrow(df_val), val_beh=mean(beh,na.rm=T) )
  rs_tst <- group_by(df_tst, binRank) %>% dplyr::summarise( tst_f=n()/nrow(df_tst) )
  
  df_summarized <- left_join(left_join(left_join(select(binning, binRank, val), 
                                                 rs_dev, by="binRank"),
                                       rs_val, by="binRank"),
                             rs_tst, by="binRank")
  
  cat("Summary symbin",fieldName,fill=T)
  print(df_summarized)
  
  df_summarized <- filter(df_summarized, binRank < 100) ## just for viewing
  
  # Barchart with frequencies
  df_plot1 <- gather(df_summarized, dataset, frequency, dev_f, val_f, tst_f)
  df_plot1$val <- reorder(as.character(factor(df_plot1$val)),df_plot1$binRank)
  try({
    plot1 <- ggplot(df_plot1, 
                    aes(x=val, y=frequency, fill=dataset))+
      geom_bar(stat="identity",position="dodge")+
      xlab(paste(fieldName,"(symbin)"))+
      theme(axis.text.x = element_text(angle = 45, hjust = 1))+
      scale_y_continuous(labels=percent)
    print(plot1)
    if (!is.null(plotFolder)) {
      ggsave(paste(plotFolder,'/plot_',fieldName,'_bin_freq.png',sep=""))
    }
  })
  
  # Linegraph with average outcomes
  df_plot2 <- gather(df_summarized, dataset, outcome, avgoutcome, val_beh)
  df_plot2$val <- reorder(as.character(factor(df_plot2$val)),df_plot2$binRank)
  try({
    plot2 <- ggplot(df_plot2, 
                    aes(x=val, y=outcome, colour=dataset, group=dataset))+
      geom_line()+geom_point()+
      xlab(paste(fieldName,"(symbin)"))+
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    print(plot2)
    if (!is.null(plotFolder)) {
      ggsave(paste(plotFolder,'/plot_',fieldName,'_bin_beh.png',sep=""))
    }
  })
  
  return (df_summarized)
}

# Create equi-weight numeric binning. The 'boundary' is the inclusive lower
# bound of the interval. Returns a dataframe like this:
#
#   boundary cases avgoutcome freq binIndex binRank   Interpretation
# 1        6     2  0.5000000  0.2        1       1    < 8
# 2        8     3  0.6666667  0.3        2       3    [8, 10>
# 3       10     5  0.6000000  0.5        3       2    >= 10
# 4       12     0        NaN  0.0        4       4    Missing values

# TODO: allow for extra param with special values passed in: add column for 'constants'
createNumbin <- function(val, outcome, minSize=0)
{
  suppressWarnings(boundaries <- cut2(val, m=minSize, onlycuts=T)) # moans about Inf otherwise
  binning <- data.frame(boundaries,stringsAsFactors=F)
  names(binning) <- c('boundary')

  valBinIndex <- applyNumbin.internal(binning, val)
  for (i in 1:nrow(binning)) {
    binning$cases[i] <- sum(valBinIndex == i)
    binning$avgoutcome[i] <- mean(outcome[valBinIndex == i],na.rm=T)
  }

  binning$avgoutcome[is.na(binning$avgoutcome)] <- NA # get rid of NaN, replace by NA
  
  # set extra columns
  binning$freq <- binning$cases/sum(binning$cases)
  binning$binIndex <- 1:nrow(binning)
  binning$binRank <- rank(binning$avgoutcome, ties.method= "first")
  return(binning)
}

applyNumbin.internal <- function(b, vec) {
  result <- findInterval(vec, b$boundary, all.inside=T)
  result[ is.na(vec) ] <- nrow(b) # TODO would do similar for 'special' values
  return(result)
}

applyNumbin <- function(b, vec) {
  return (b$avgoutcome[applyNumbin.internal(b, vec)])
#   return (b$binRank[applyNumbin.internal(b, vec)])
}

plotNumbin <- function(binz, plotFolder=NULL) 
{
  print(binz)
  # Barchart with frequencies
  binz$interval <- factor(binz$interval, levels=binz$interval)
  df_plot1 <- gather(binz, dataset, frequency, dev_f, val_f, tst_f)
  try({
    plot1 <- ggplot(df_plot1, 
                    aes(x=interval, y=frequency, fill=dataset))+
      geom_bar(stat="identity",position="dodge")+
      xlab(fieldName)+
      theme(axis.text.x = element_text(angle = 45, hjust = 1))+
      scale_y_continuous(labels=percent)
    print(plot1)
    if (!is.null(plotFolder)) {
      ggsave(paste(trainDataFolder,'/plot_',fieldName,'_nb_freq.png',sep=""))
    }
  })
  
  # Linegraph with average outcomes
  df_plot2 <- gather(binz, dataset, outcome, avgoutcome, val_beh)
  try({
    plot2 <- ggplot(df_plot2, 
                    aes(x=interval, y=outcome, colour=dataset, group=dataset))+
      geom_line()+geom_point()+
      xlab(fieldName)+
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    print(plot2)
    if (!is.null(plotFolder)) {
      ggsave(paste(plotFolder,'/plot_',fieldName,'_nb_beh.png',sep=""))
    }
  })
}

# replace NAs in data set with mean - should do knnImpute
imputeNAs <- function(ds, verbose=F) {
  ds <- as.data.frame(ds)
  cat("Imputing",sum(!complete.cases(ds)),"rows from total of",nrow(ds), fill=TRUE)
  for (colNo in which(sapply(ds, is.numeric))) {
    aCol <- select(ds, colNo)
    if (any(is.na(aCol))) {
      m <- colMeans(aCol,  na.rm = TRUE)
      if (verbose) {
        cat("   imputing", colnames(ds) [colNo], sum(!complete.cases(aCol)),"NAs with mean", m, fill=TRUE)
      }
      ds[!complete.cases(aCol), colNo] <- m
    }
  }
  return(ds)
}

# Kaggle's LogLoss evaluation function
logLoss <- function(act, pred)
{
  eps <- 1e-15
  if (!is.logical(act)) {
    stop("Logloss expects a logical as first argument")
  }
  ll <- -1*sum(act*log(pmax(eps,pred)) + (1-act)*log(pmax(eps,1-pred)))/length(pred)
  return(ll)
}

# check if column values are all dates
isDate <- function(vec) { 
  all( grepl( "^\\d{2}[A-Z]{3}\\d{2}", vec[nzchar(vec)]) ) # check date fmt "12OCT13" (or empty)
}

# check if column values are all booleans
isBoolean <- function(vec) { 
  all( grepl( "^true$|^false$", vec[nzchar(vec)]) )
}

# DA for one vector - assuming the dataframes all contain only one field
dataAnalysisOne <- function(dfDev, dfVal, dfTest, fldName, 
                            generatePlots=F, plotFolder=NULL)
{
  dsFull <- rbind(dfDev, dfVal)
  
  dataMetrics <- nearZeroVar(dsFull, saveMetrics=TRUE)
  dataMetrics$className <- lapply(dsFull,class)
  dataMetrics$isSymbolic <- dataMetrics$className %in% c("factor", "character")
  dataMetrics$isNumeric <- dataMetrics$className %in% c("integer", "numeric", "logical")
  dataMetrics$nDistinct <- lapply(dsFull, function(col) { return (length(unique(col))) })
  # instead of dataMetrics$percentUnique * nrow(dsFull) / 100, to account for NAs as well
  dataMetrics$nNA <- sapply(dsFull, function(vec) { return (sum(is.na(vec))) })
  symbolicFldNames <- rownames(dataMetrics) [dataMetrics$isSymbolic]
  
  if (length(symbolicFldNames) > 0) {
    dataMetrics$isDate <- rownames(dataMetrics) %in% symbolicFldNames[ sapply(symbolicFldNames, function(colName) { isDate(na.omit(dsFull[[colName]])) } ) ]
    dataMetrics$isBoolean <- rownames(dataMetrics) %in% symbolicFldNames[ sapply(symbolicFldNames, function(colName) { isBoolean(na.omit(dsFull[[colName]])) } ) ]
  } else {
    dataMetrics$isDate <- F
    dataMetrics$isBoolean <- F
  }    
  
  # Get AUC estimates for all predictors
  dataMetrics$Overlap <- NA
  dataMetrics$ksTest <- NA
  
  if (nrow(dataMetrics) != 1) {
    print(dataMetrics)
    stop("STOP: expected one row in data metrics frame")
  }
  
  u1 <- unique(dsFull[[1]])
  u2 <- unique(dfTest[[1]])
  dataMetrics$Overlap[1] <- length(intersect(u1,u2)) / length(union(u1,u2))
  
  if (dataMetrics$isNumeric[1] && dataMetrics$nDistinct[1] > 1) {
    # K-S test for similarity test/train distributions and checking overlap distincts test/train
    
    ksMetric <- suppressWarnings(ks.test(
      density(dsFull[[1]], na.rm=T)[["y"]], 
      density(dfTest[[1]], na.rm=T)[["y"]]) [["statistic"]])
    if (generatePlots) {
      try({
        plotFrame <- data.frame(c(dfDev[[1]], dfVal[[1]], dfTest[[1]]), 
                                factor(c(rep("dev",nrow(dfDev)),
                                         rep("val",nrow(dfVal)),
                                         rep("test",nrow(dfTest)))))
        names(plotFrame) <- c('values','dataset')
        densityPlot <-
          ggplot(plotFrame, aes(x=values)) + geom_density(aes(group=dataset, colour=dataset))+
            xlab(paste(fldName, "K-S score:", ksMetric, 
                       " Overlap:", dataMetrics$Overlap[1]))
        print(densityPlot)
        if (!is.null(plotFolder)) {
          ggsave(paste(plotFolder,'/plot_',fldName,'_density.png',sep=""))
        }
      })
    }
    
    dataMetrics$ksTest[1] <- ksMetric
  
    if (F) {
      # identify outliers (Ivar special code)
      freq <- table(dsFull[[1]])
      minObs  <- 100 # TODO: tune
      q <- quantile(freq, probs=c(0.25,0.75))
      limit <- q[2] + 3 * (q[2]-q[1]) # use the interquartile range for outliers
      limit <- max(limit, minObs) # at least nObs
      special <- freq[freq > limit]
      specialTot <- sum(special)
      cat("limit:", limit, "** special:", names(special), "freqs", special, 
          "tot:", specialTot, "\n")
    }    
  }
  
  return(dataMetrics)
}

# Data analysis on a train set (already split in dev/val)
dataAnalysis <- function(dfDev, dfVal, dfTest, generatePlots=F, plotFolder=NULL)
{
  metrics <- NULL
  for (fldName in names(dfTest)) {
    cat("Basic DA for: ", fldName, fill=T)
    metricOne <- dataAnalysisOne(dfDev[fldName], dfVal[fldName], dfTest[fldName], # data frames
                                 fldName, 
                                 generatePlots, plotFolder)
    #print(metricOne)
    if (is.null(metrics)) {
      metrics <- metricOne
    } else {
      metrics <- rbind(metrics, metricOne)
    }
  }
  return(metrics)
}
