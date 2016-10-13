# LASSO_cocaine_frontiers_multipleTestSets.R
# Classifying individuals with cocaine dependence using a penalized regression
# approach (LASSO). 
# This code runs LASSO on 1,000 (default) different sets training and test sets
# ** Instructions **
# Copy all codes (including this file) and a data file to a folder
# & Set your working directory to the folder in Line 34 
# Carefully check Lines 34-46 and make necessary changes
# 
# If you used (some of) this code for your work, consider citing
# Ahn*, Ramesh*, Moeller, & Vassileva (2016) Frontiers in Psychopathology.
# This code will generate Figure 1 in Ahn, Ramesh et al. (2016)
# Programmed by Woo-Young Ahn
# Replace quartz() with x11() if you run this code in a Windows or a Linux computer.
# **Warning** It will take apprxoimately 3 hr to run this code with default settings.

rm(list=ls())      # clear workspace

# Check if required packages are installed (from John Kruschke)
# If not, install them
reqPackages = c("glmnet", "ggplot2", "pROC", "corrplot")
have = reqPackages %in% rownames(installed.packages())
if ( any(!have) ) { 
  cat("** Some required packages cannot be found. They are being installed now **\n")
  install.packages( reqPackages[!have] ) 
}

library(glmnet)   # penalized regression
library(ggplot2)  # paper-ready figures
library(pROC)     # ROC curves
library(corrplot) # Correlation plots
##############################################################################
# Variables to customize
wd_path = "~/Dropbox/CARI_behavior/"  # working directory. Put all files in this folder.
outOfSample = T              # Always TRUE. Split data into independent training and test sets? For this code. 
dat_path = file.path(wd_path, "cocaineData_frontiers.txt")   # path for a data file
minAlpha = 1                 # Default=1. Alpha=1 for LASSO
nFolds  = 5                  # Default=5. k-fold
numIterations = 100          # Default=100. Number of iterations
numDifferentDivisions = 1000 # Default=1000. Number of different division of test/training sets
survivalRate_cutoff = 0.05   # Default=0.05. Cutoff for survival rate.
excludeAge = F               # Default=F. Remove age variable? T(rue) or F(alse)
set.seed(43210)              # Comment this line not to set a seed.
                             # (Optional) To check reproducibility. 
                             # Default=43210. With this seed and all default settings,
                             # mean AUC should be 0.975 (training set) and 0.892 (test set)
##############################################################################
cat("Generating out-of-sample predictions? ", outOfSample, "\n")
cat("**Warning** It can take a long time (~3 hr with default settings. Change values in lines 39-40 to save time.\n")
setwd(wd_path)  # set (go to) working directory
source("cor.mtest.R")  # from the help file of 'corrplot'
source("sample_equalProp.R") # To set the proportion of two groups equal in training and test sets 

# Read (raw) data
rawDat = read.table( dat_path, header=T)
# remove subject ID from the dataset
rawDat = subset( rawDat, select = -c(subject) )

# exclude age regressor?
if (excludeAge) {
  rawDat = subset( rawDat, select = -c(AGE) )
}

# addictVar --> dependent variable. 0 = healthy control (HC), 1 = cocaine user
addictVar = as.integer( rawDat$DIAGNOSIS )  
# indepDat --> independent variables 
indepDat = subset(rawDat, select = -c(DIAGNOSIS)) 

# categorical variables --> no z-scoring. In this dataset, 'male' (1 or 0) is the only categorical variable.
male = indepDat[, "Male"]  
# contDat --> a matrix only with continuous variables
contDat = subset(indepDat, select = -c(Male) ) # remove 'Male' before z-scoring continuous variables

# z-score continuous variables
contDat = scale(contDat)
# combine raw categorical and z-scored continuous (independent) variables
allDat = as.matrix( data.frame(male, contDat) )

numSubjs = length(addictVar) # number of participants (i.e., n)
numPredictors = dim(allDat)[2] + 1  # number of features (i.e, p). +1 because of intercept

# lassoDat --> a matrix: 1st column=dependent variable, the other columns=independent variables
lassoDat = cbind(addictVar, allDat) 

# to store mean AUC values
all_auc_min_trainSet = vector()
all_auc_min_testSet = vector()

# A text based progress bar
progressBar = txtProgressBar(min=1, max=numDifferentDivisions, style=3)
cat("Running LASSO w/ ", numDifferentDivisions, " differnt combinations of training and test sets. \n")

for (pIdx in 1:numDifferentDivisions) {
  
  sampledDat = sample_equalProp(lassoDat)
  
  # '_t' --> training set
  # '_v' --> test (validation) set
  
  # training set
  trainDat = sampledDat[[1]]                                     # matrix including the dependent variable   
  addictVar_t = trainDat[, "addictVar"]                          # dependent variable in the training set
  trainVar = trainDat[, -which(colnames(trainDat)=="addictVar")] # matrix without the dependent variable
  #yVar = addictVar_t   # exactly same as addictVar_t ....
  
  # test set
  predDat = sampledDat[[2]]                                   # matrix including the dependent variable   
  addictVar_v = predDat[, "addictVar"]                        # dependent variable in the test set
  predVar = predDat[, -which(colnames(predDat)=="addictVar")] # matrix without the dependent variable
  # number of participants in each set
  numSubjs_t = length(addictVar_t)
  numSubjs_v = length(addictVar_v)
  
  #####################################################
  ### Implement LASSO                               ###
  #####################################################
  
  # To save values for the test (validation) set.
  # w/ min lambda
  all_predictedVar_min = array(NA, c(numSubjs_v, numIterations) )
  all_beta_min = array(NA, c(numPredictors, numIterations) )
  all_auc_min = vector()
  all_survivalRate_min = array(NA, c(numPredictors, numIterations) )
  # w/ +1se labmda
  all_predictedVar_1se = array(NA, c(numSubjs_v, numIterations) )
  all_beta_1se = array(NA, c(numPredictors, numIterations) )
  all_auc_1se = vector()
  all_survivalRate_1se = array(NA, c(numPredictors, numIterations) )
  
  # To save values for the train set (w/ min lambda)
  all_predictedVar_min_t = array(NA, c(numSubjs_t, numIterations) )
  all_auc_min_t = vector()
  
  # fit LASSO with the training set
  for (rIdx in 1:numIterations) {
    lasso_glmnet_best = glmnet(x=trainVar, y=addictVar_t, family="binomial", standardize=F, alpha=minAlpha, nlambda=200, maxit=10^6)  
    lasso_cv_glmnet_best = cv.glmnet(x=trainVar, y=addictVar_t, family="binomial", standardize=F, alpha=minAlpha, nlambda=200, nfolds=nFolds, maxit=10^6)
    
    # test set
    tmp_predAddictVar_min = predict(lasso_glmnet_best, newx = predVar, s = lasso_cv_glmnet_best$lambda.min , type="response")
    tmp_auc_min = roc(addictVar_v, as.numeric(tmp_predAddictVar_min))
    
    # training set
    tmp_predAddictVar_min_t = predict(lasso_glmnet_best, newx = trainVar, s = lasso_cv_glmnet_best$lambda.min, type="response")
    tmp_auc_min_t = roc(addictVar_t, as.numeric(tmp_predAddictVar_min_t))
    
    tmp_predAddictVar_1se = predict(lasso_glmnet_best, newx = predVar, s = lasso_cv_glmnet_best$lambda.1se, type="link" )
    tmp_auc_1se = roc(addictVar_v, as.numeric(tmp_predAddictVar_1se))
    
    tmp_beta = predict(lasso_glmnet_best, s = lasso_cv_glmnet_best$lambda.min, type="coefficient" )  # uisng min labmda
    tmp_beta_1se = predict(lasso_glmnet_best, s = lasso_cv_glmnet_best$lambda.1se, type="coefficient" )  # uisng 1se labmda
    
    all_predictedVar_min[, rIdx] = tmp_predAddictVar_min
    all_beta_min[, rIdx] = as.matrix(tmp_beta)
    all_auc_min[rIdx] = as.numeric(tmp_auc_min$auc)
    all_survivalRate_min[, rIdx] = as.numeric(abs(tmp_beta) > 0)
    
    # 1se
    all_predictedVar_1se[, rIdx] = tmp_predAddictVar_1se
    all_beta_1se[, rIdx] = as.matrix(tmp_beta_1se)
    all_auc_1se[rIdx] = as.numeric(tmp_auc_1se$auc)
    all_survivalRate_1se[, rIdx] = as.numeric(abs(tmp_beta_1se) > 0)
    
    # train set
    all_predictedVar_min_t[, rIdx] = tmp_predAddictVar_min_t
    all_auc_min_t[rIdx] = as.numeric(tmp_auc_min_t$auc)
    
  }
  
  ##############################################################
  ### compute mean values of 100 iterations (on each pIdx)   ###
  ##############################################################
  
  # test set
  predAddictVar_min = apply(all_predictedVar_min, 1, mean)
  auc_min = mean(all_auc_min, na.rm=T)
  # training set
  predAddictVar_min_t = apply(all_predictedVar_min_t, 1, mean)
  auc_min_t = mean(all_auc_min_t, na.rm=T)
  
  all_auc_min_testSet[pIdx] = auc_min
  all_auc_min_trainSet[pIdx] = auc_min_t
  
  # ggplot2 (USING min lambda)
  dat = data.frame(Actual = as.numeric(addictVar_v), Predicted = as.numeric(predAddictVar_min))
  
  # 1se
  predAddictVar_1se = apply(all_predictedVar_1se, 1, mean)
  auc_1se = mean(all_auc_1se, na.rm=T)
  
  dat_1se = data.frame(Actual = as.numeric(addictVar_v), Predicted = as.numeric(predAddictVar_1se))
  # test set 
  #dat_t = data.frame(Actual = as.numeric(addictVar_t), Predicted = as.numeric(predAddictVar_min_t), Group=as.factor(groupMemb[tSeq]))
  dat_t = data.frame(Actual = as.numeric(addictVar_t), Predicted = as.numeric(predAddictVar_min_t))
  
  ### Drawing
  auc_figure_tmp = roc(Actual ~ Predicted, data = dat)
  auc_figure = as.numeric( auc_figure_tmp$auc )
  auc_figure_digit = prettyNum(auc_figure, digits=3, nsmall=3,width=5, format="fg")
  auc_dat = data.frame(Sens = auc_figure_tmp$sensitivities, OneMinusSpec = 1 - auc_figure_tmp$specificities)
  
  auc_figure_tmp = roc(Actual ~ Predicted, data = dat_1se)
  auc_figure = as.numeric( auc_figure_tmp$auc )
  auc_figure_digit = prettyNum(auc_figure, digits=3, nsmall=3, width=5, format="fg")
  auc_dat = data.frame(Sens = auc_figure_tmp$sensitivities, OneMinusSpec = 1 - auc_figure_tmp$specificities)
  
  # training set
  auc_figure_tmp = roc(Actual ~ Predicted, data = dat_t)
  auc_figure = as.numeric( auc_figure_tmp$auc )
  auc_figure_digit = prettyNum(auc_figure, digits=3, nsmall=3, width=5, format="fg")
  auc_dat = data.frame(Sens = auc_figure_tmp$sensitivities, OneMinusSpec = 1 - auc_figure_tmp$specificities)
  
  # survival rate
  mean_survivalRate = apply(all_survivalRate_min, 1, mean)
  mean_survivalRate_cutoff = ( mean_survivalRate > survivalRate_cutoff ) * apply(all_beta_min, 1, mean)
  # betas of regressors
  beta_min = data.frame( mean = apply(all_beta_min, 1, mean), sd =  apply(all_beta_min, 1, sd) )
  rownames(beta_min) = rownames(tmp_beta)
  beta_min_cutoff = data.frame( mean = mean_survivalRate_cutoff, sd =  apply(all_beta_min, 1, sd), survival =  mean_survivalRate)
  rownames(beta_min_cutoff) = rownames(tmp_beta)
  
  # 1se
  mean_survivalRate_1se = apply(all_survivalRate_1se, 1, mean)
  mean_survivalRate_cutoff_1se = ( mean_survivalRate_1se > survivalRate_cutoff ) * apply(all_beta_1se, 1, mean)
  # betas of regressors
  beta_1se = data.frame( mean = apply(all_beta_1se, 1, mean), sd =  apply(all_beta_1se, 1, sd) )
  rownames(beta_1se) = rownames(tmp_beta_1se)
  beta_1se_cutoff = data.frame( mean = mean_survivalRate_cutoff_1se, sd =  apply(all_beta_1se, 1, sd), survival =  mean_survivalRate_1se)
  rownames(beta_1se_cutoff) = rownames(tmp_beta_1se)
  
  #cat( "Permutation # ", pIdx, "out of ", numDifferentDivisions , " is done.\n")
  
  setTxtProgressBar(progressBar, pIdx)
}

hist(all_auc_min_trainSet, xlim=c(0,1))
hist(all_auc_min_testSet, xlim=c(0,1))

hist_dat = data.frame(auc_train = all_auc_min_trainSet, auc_test = all_auc_min_testSet)

# Distribution of AUCs (train sets)
quartz()
ggplot(hist_dat, aes(x=auc_train)) +
  xlim(0,1) +
  geom_histogram(colour="dark grey", binwidth = 0.02, fill="white") +
  geom_vline(aes(xintercept=mean(auc_train, na.rm=T)),   # Ignore NA values for mean
             color="black", linetype="dashed", size=1) +
  ggtitle("Distribution of AUCs (Training Sets)") +
  theme(plot.title=element_text(size=20)) +
  theme(axis.title = element_text(size = 20) ) + 
  theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color  
  xlab("AUC") + ylab("Frequency")

# Distribution of AUCs (test sets)
quartz()
ggplot(hist_dat, aes(x=auc_test)) +
  xlim(0,1) +
  geom_histogram(colour="dark grey", binwidth = 0.02, fill="white") +
  geom_vline(aes(xintercept=mean(auc_test, na.rm=T)),   # Ignore NA values for mean
             color="black", linetype="dashed", size=1) +
  ggtitle("Distribution of AUCs (Test Sets) ") +
  theme(plot.title=element_text(size=20)) +
  theme(axis.title = element_text(size = 20) ) + 
  theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color  
  xlab("AUC") + ylab("Frequency")

cat("\n All done! \n")
