# LASSO_cocaine_frontiers.R
# Classifying individuals with cocaine dependence using a penalized regression
# approach (LASSO). 
#
# ** Instructions **
# Copy all codes (including this file and cor.mtest.R) and a data file to a folder
# & Set your working directory to the folder in Line 34 
# Carefully check Lines 34-48 and make necessary changes
# 
# If you used (some of) this code for your work, consider citing
# Ahn*, Ramesh*, Moeller, & Vassileva (2016) Frontiers in Psychopathology.
# This code will generate Figure 1 in Ahn, Ramesh et al. (2016)
# Programmed by Woo-Young Ahn
# Replace quartz() with x11() if you run this code in a Windows or a Linux computer.

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
outOfSample = F              # Split data into independent training and test sets? T(rue) or F(alse). 
splitBy = 3                  # Default=3. Use 1/splitBy (=33%) of data as a test set
whichSeq = 1                 # Default=1. Which sequence to use for a test set? 1, 2, or 3
dat_path = file.path(wd_path, "cocaineData_frontiers.txt")   # path for a data file
minAlpha = 1                 # Default=1. Alpha=1 for LASSO
nFolds  = 5                  # Default=5. k-fold
numIterations = 1000          # Default=1000. Number of iterations
survivalRate_cutoff = 0.05   # Default=0.05. Cutoff for survival rate.
plotColor = c("#E41A1C")     # Use your favorite color for your ROC curve
excludeAge = F               # Default=F. Remove age variable? T(rue) or F(alse)
set.seed(43210)              # Comment this line not to set a seed.
                             # (Optional) To check reproducibility. 
                             # Default=43210. With this seed and all default settings,
                             # AUC should be 0.952 (training set) and 0.912 (test set)
##############################################################################
cat("Generating out-of-sample predictions? ", outOfSample, "\n")

setwd(wd_path)  # set (go to) working directory
source("cor.mtest.R")  # from the help file of the 'corrplot' package

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

#####################################################
### Divided lassoDat into trainDat and testDat    ###
#####################################################
## 3 sequences
## Decide which one to use for prediction

allSeq = 1:numSubjs
subjSeq1 = seq(whichSeq, numSubjs, by = splitBy)  # e.g., 1, 4, 7, ..., 82 --> N = 28

if (outOfSample) { # if yes, use 2/3 of data as the training set and 1/3 of data as the test (validation) set
  vSeq = subjSeq1
  tSeq = allSeq[-vSeq]
} else {  # then, use all data as the training and test sets
  vSeq = allSeq
  tSeq = allSeq
}

# validation(test) set
testDat = lassoDat[ vSeq, ]  # matrix including the dependent variable   
testVar = testDat[, -which(colnames(testDat)=="addictVar")] # matrix without the dependent variable
# train set
trainDat = lassoDat[ tSeq, ] # matrix including the dependent variable      
trainVar = trainDat[, -which(colnames(trainDat)=="addictVar")] # matrix without the dependent variable

# dependent variable in training and test sets
addictVar_t = addictVar[tSeq]  # '_t' --> training set
addictVar_v = addictVar[vSeq]  # '_v' --> test (validation) set
# number of participants in each set
numSubjs_t = length(addictVar_t)
numSubjs_v = length(addictVar_v)

#####################################################
### Implement LASSO                               ###
#####################################################

# To save values for the test (validation) set.
# for min lambda
all_predictedVar_min = array(NA, c(numSubjs_v, numIterations) )    # predicted addictVar (on the test set)
all_beta_min = array(NA, c(numPredictors, numIterations) )         # fitted beta coefficients (w/ train set)
all_auc_min = vector()                                             # AUC values (on the test set)
all_survivalRate_min = array(NA, c(numPredictors, numIterations) ) # survival rate (w/ train set)
# for +1se lambda
all_predictedVar_1se = array(NA, c(numSubjs_v, numIterations) )    # predicted addictVar (on the test set)
all_beta_1se = array(NA, c(numPredictors, numIterations) )         # fitted beta coefficients (w/ train set)
all_auc_1se = vector()                                             # AUC values (on the test set)
all_survivalRate_1se = array(NA, c(numPredictors, numIterations) ) # survival rate (w/ train set)

# To save values for the train set (w/ min lambda)
all_predictedVar_min_t = array(NA, c(numSubjs_t, numIterations) )  # predicted addictVar (on the train set)
all_auc_min_t = vector()                                           # AUC values (on the test set)

# A text based progress bar
progressBar = txtProgressBar(min=1, max=numIterations, style=3)
cat("Running ", numIterations, " iterations.\n")

for (rIdx in 1:numIterations) {
  # fit LASSO with the training set
  lasso_glmnet = glmnet(x=trainVar, y=addictVar_t, family="binomial", standardize=F, alpha=minAlpha, maxit=10^6)  
  lasso_cv_glmnet = cv.glmnet(x=trainVar, y=addictVar_t, family="binomial", standardize=F, alpha=minAlpha, nfolds=nFolds, maxit=10^6)
  
  ## test predictions on the test set (with min lambda)
  tmp_predAddictVar_min = predict(lasso_glmnet, newx = testVar, s = lasso_cv_glmnet$lambda.min , type="response")
  # compute AUC of the ROC curve
  tmp_auc_min = roc(addictVar_v, as.numeric(tmp_predAddictVar_min) )
  
  # test predictions on the training set (with min lambda)
  tmp_predAddictVar_min_t = predict(lasso_glmnet, newx = trainVar, s = lasso_cv_glmnet$lambda.min, type="response")
  tmp_auc_min_t = roc(addictVar_t, as.numeric(tmp_predAddictVar_min_t) )
  
  ## test predictions on the test set (with +1se lambda)
  tmp_predAddictVar_1se = predict(lasso_glmnet, newx = testVar, s = lasso_cv_glmnet$lambda.1se, type="link" )
  tmp_auc_1se = roc(addictVar_v, as.numeric(tmp_predAddictVar_1se) )
  
  # extract beta coefficients with min lambda  
  tmp_beta = predict(lasso_glmnet, s = lasso_cv_glmnet$lambda.min, type="coefficient" ) 
  # extract beta coefficients with min lambda
  tmp_beta_1se = predict(lasso_glmnet, s = lasso_cv_glmnet$lambda.1se, type="coefficient" )
  
  # save predictions made on the test set (w/ min lambda)
  all_predictedVar_min[, rIdx] = tmp_predAddictVar_min
  all_beta_min[, rIdx] = as.matrix(tmp_beta)
  all_auc_min[rIdx] = as.numeric(tmp_auc_min$auc)
  all_survivalRate_min[, rIdx] = as.numeric(abs(tmp_beta) > 0)
  
  # save predictions made on the test set (w/ +1se lambda)
  all_predictedVar_1se[, rIdx] = tmp_predAddictVar_1se
  all_beta_1se[, rIdx] = as.matrix(tmp_beta_1se)
  all_auc_1se[rIdx] = as.numeric(tmp_auc_1se$auc)
  all_survivalRate_1se[, rIdx] = as.numeric(abs(tmp_beta_1se) > 0)
  
  # save predictions made on the train set (w/ min lambda)
  all_predictedVar_min_t[, rIdx] = tmp_predAddictVar_min_t
  all_auc_min_t[rIdx] = as.numeric(tmp_auc_min_t$auc)
  
  #cat( "Iteration ", rIdx, "out of ", numIterations, " is finished\n")
  setTxtProgressBar(progressBar, rIdx)
}

#################################################
### compute mean values of 1,000 iterations ###
#################################################

# predicted addictVar on the test set (w/ min lambda)
predAddictVar_min = apply(all_predictedVar_min, 1, mean)
# average AUC on the test set (w/ min lambda)
auc_min = mean(all_auc_min, na.rm=T)

# predicted addictVar on the training set (w/ min lambda)
predAddictVar_min_t = apply(all_predictedVar_min_t, 1, mean)
# average AUC on the training set (w/ min lambda)
auc_min_t = mean(all_auc_min_t, na.rm=T)

## In case you want to draw a ROC curve generated by +1se, use these lines..
# predicted addictVar on the test set (w/ +1se lambda)
#predAddictVar_1se = apply(all_predictedVar_1se, 1, mean)
# average AUC on the test set (w/ +1se lambda)
#auc_1se = mean(all_auc_1se, na.rm=T)

# To plot ROC curves (test set) w/ ggplot2 
dat_min = data.frame(Actual = as.numeric(addictVar_v), Predicted = as.numeric(predAddictVar_min))
# To plot ROC curves (train set) w/ ggplot2 
dat_min_t = data.frame(Actual = as.numeric(addictVar_t), Predicted = as.numeric(predAddictVar_min_t))

### Drawing
auc_figure_tmp = roc(Actual ~ Predicted, data = dat_min)
auc_figure = as.numeric( auc_figure_tmp$auc )
auc_figure_digit = prettyNum(auc_figure, digits=3, nsmall=3,width=5, format="fg")
auc_dat = data.frame(Sens = auc_figure_tmp$sensitivities, OneMinusSpec = 1 - auc_figure_tmp$specificities)

# Draw a ROC curve (test set)
quartz() 
h1 = ggplot(auc_dat, aes(x=OneMinusSpec, y=Sens)) + 
  geom_path(alpha=1, size=1, colour = plotColor) +
  ggtitle("ROC Curve (Test Set)") +
  annotate("text", label = paste("AUC = ", auc_figure_digit, sep=""), x = 0.6, y = 0.1, size = 15, colour = "black") +
  theme(plot.title=element_text(size=30)) +
  theme(axis.title = element_text(size = 30) ) + 
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1) , linetype="dashed") +
  theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color  
  xlab("1 - Specificity") + ylab("Sensitivity")
print(h1)

# training set
auc_figure_tmp_t = roc(Actual ~ Predicted, data = dat_min_t)
auc_figure_t = as.numeric( auc_figure_tmp_t$auc )
auc_figure_digit_t = prettyNum(auc_figure_t, digits=3, nsmall=3, width=5, format="fg")
auc_dat_t = data.frame(Sens = auc_figure_tmp_t$sensitivities, OneMinusSpec = 1 - auc_figure_tmp_t$specificities)

quartz()
h2 = ggplot(auc_dat_t, aes(x=OneMinusSpec, y=Sens)) + 
  geom_path(alpha=1, size=1, colour = plotColor) +
  ggtitle("ROC Curve (Training Set)") +
  annotate("text", label = paste("AUC = ", auc_figure_digit_t, sep=""), x = 0.6, y = 0.1, size = 15, colour = "black") +
  theme(plot.title=element_text(size=30)) +
  theme(axis.title = element_text(size = 30) ) + 
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1) , linetype="dashed") +
  theme(axis.text = element_text(size = 20, colour="black")) +   # for black tick label color  
  xlab("1 - Specificity") + ylab("Sensitivity")
print(h2)


# If all samples were used (outOfSample == F), run the following lines:
if (outOfSample==F) {

  ##########################################
  ### plot beta regressors               ###
  ### For this, set outOfSample = F      ###
  ##########################################
  
  # Calculate survival rate  (w/ min lambda)
  mean_survivalRate = apply(all_survivalRate_min, 1, mean)
  # if survival rate < cutoff (5% = 0.05), set its mean to zero
  mean_survivalRate_cutoff = ( mean_survivalRate > survivalRate_cutoff ) * apply(all_beta_min, 1, mean)
  
  # beta coefficients of regressors (w/ min lambda)
  # beta_min 
  bounds_min = apply(all_beta_min, 1, quantile, probs = c(0.025, 0.975))  # 95% confidence interval
  rownames(bounds_min) = c("lb", "ub")
  beta_min = data.frame(mean=apply(all_beta_min, 1, mean), lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate)
  rownames(beta_min) = rownames(tmp_beta)
  # beta_min_cutoff --> remove variables w/ less than 5% survival rate
  beta_min_cutoff = data.frame(mean=mean_survivalRate_cutoff, lb = bounds_min["lb",], ub = bounds_min["ub", ], survival=mean_survivalRate)
  rownames(beta_min_cutoff) = rownames(tmp_beta)
  
  cocaReg = beta_min_cutoff
  cocaReg$predictor = rownames(cocaReg)
  
  # mark sig. ones
  cocaReg$sig = as.numeric(cocaReg$lb * cocaReg$ub > 0)  # sig <-- 95% CI excludes zero
  
  ## mark colors
  # beta=zero --> 0, not zero but non-sig --> 1, sig --> 2
  cocaReg$dotColor1 = ifelse(cocaReg$mean ==0, 0, 1)
  cocaReg$dotColor2 = ifelse(cocaReg$dotColor1 > 0 & cocaReg$sig > 0, 2, 1)
  cocaReg$dotColor = as.factor(cocaReg$dotColor1 * cocaReg$dotColor2)
  
  # if there are less than 3 types of dotColors (i.e., all variables' 95% CI excludes 0),
  # only use two colors. Otherwise, use three colors
  if (length( unique(cocaReg$dotColor)) > 2) {
    plotCols = c(rgb(0,0,0,0.0), rgb(1,0,0,0.5), rgb(1,0,0,0.8) )
  } else {
    plotCols = c(rgb(0,0,0,0.0), rgb(1,0,0,0.8) )
  }
  
  # remove intercept
  cocaReg = cocaReg[-1, ]
  
  # plot cocaReg
  plotVars = c("Male","AGE","EDU_YRS",
               "BIS_attention", "BIS_motor", "BIS_NonPL", 
               "Stop_ssrt", "IMT_OMIS_errors", "IMT_COMM_errors", "A_IMT",  "B_D_IMT", 
               "lnk_adjdd", "lkhat_Kirby", "REVLR_per_errors", 
               "IGT_TOTAL")
  plotLabels = c("Sex","Age","Education",
                 "BIS Attn", "BIS Motor", "BIS Nonpl", 
                 "SSRT", "IMT FN", "IMT FP", "IMT Discriminability", "IMT Response bias",
                 "ln(k)", "ln(k), Kirby", "PRL perseverance", 
                 "IGT Score"   )
  quartz()
  h_coca = ggplot(cocaReg, aes(x=predictor, y=mean, colour = dotColor)) + coord_flip() +
    geom_errorbar(aes(ymin=lb, ymax=ub), width=.1) +
    geom_line() +
    geom_point() +
    theme(text = element_text(size=9), legend.position="none", axis.text.y = element_text(colour = cocaReg$labColor)) +
    scale_color_manual(values = plotCols) +  # colors for "no" and "yes" (alphabetical order...)
    scale_x_discrete(limits=plotVars, labels = plotLabels ) +
    xlab("Predictors") +
    ylab("Beta estimates") +
    ggtitle("Predicting Cocaine Group Membership") +
    theme(axis.text.x = element_text(size = 18), axis.text.y = element_text(size = 14), axis.title = element_text(size=20))
  print(h_coca)
  
  ##########################################
  ### Figure s1                          ###
  ### For this, set outOfSample = F      ###
  ##########################################
  
  forP = allDat[, c("BIS_attention", "BIS_motor", "BIS_NonPL", "Stop_ssrt", "IMT_OMIS_errors", "IMT_COMM_errors", "A_IMT", "B_D_IMT", "lnk_adjdd", "lkhat_Kirby", "REVLR_per_errors", "IGT_TOTAL")]
  pLabels2 = c("BIS Attn", "BIS Motor", "BIS Nonpl", "SSRT", "IMT FN", "IMT FP", "IMT d'", "IMT B[D]", "ln(k)", "ln(k), Kirby", "PRL persev", "IGT Score")
  
  # http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
  col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
  M = cor(forP)
  colnames(M) = pLabels2
  rownames(M) = pLabels2
  p.mat <- cor.mtest(forP)
  
  quartz()
  corrplot(M, method="color", col=col(200), add = F,
           type="upper", order="original", outline=F,
           addgrid.col=T, 
           addCoef.col = "black", # Add coefficient of correlation
           tl.col="black", tl.srt=90, #Text label color and rotation
           # Combine with significance
           p.mat = p.mat[[1]], sig.level = 0.05, insig = "blank", 
           plotCI = "n",
           #lowCI.mat = p.mat[[2]], uppCI.mat = p.mat[[3]],
           # hide correlation coefficient on the principal diagonal
           diag=T 
  )
}

cat("\n All done! \n")
