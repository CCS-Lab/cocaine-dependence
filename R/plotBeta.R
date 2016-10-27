plotBeta = function(egOutput, sortBeta = T, removeIntercept = T) {
  # plot beta coefficients of easyGlmnet outputs
  # e.g., 
  # output = easyGlmnet(...)
  # plotBeta(output[[3]])
  
  # Detect OS 
  if ( Sys.info()["sysname"] == "Darwin" ) {  # if Mac OS X --> use 'quartz'
    x11 <- function( ... ) quartz( ... ) 
  }
  
  tmpReg = egOutput
  tmpReg$predictor = rownames(tmpReg)
  
  # mark sig. ones
  tmpReg$sig = as.numeric(tmpReg$lb * tmpReg$ub > 0)  # sig <-- 95% CI excludes zero
  
  ## mark colors
  # beta=zero --> 0, not zero but non-sig --> 1, sig --> 2
  tmpReg$dotColor1 = ifelse(tmpReg$mean ==0, 0, 1)
  tmpReg$dotColor2 = ifelse(tmpReg$dotColor1 > 0 & tmpReg$sig > 0, 2, 1)
  tmpReg$dotColor = as.factor(tmpReg$dotColor1 * tmpReg$dotColor2)
  
  # if there are less than 3 types of dotColors (i.e., all variables' 95% CI excludes 0),
  # only use two colors. Otherwise, use three colors
  if (length( unique(tmpReg$dotColor)) > 2) {
    plotCols = c(rgb(0,0,0,0.0), rgb(1,0,0,0.3), rgb(1,0,0,0.9) )
  } else {
    plotCols = c(rgb(0,0,0,0.0), rgb(1,0,0,0.9) )
  }
  
  if (removeIntercept) {
    # remove intercept
    tmpReg = tmpReg[-1, ]
  }
  
  if (sortBeta) {  # sort tmpReg with 'mean' column
    tmpReg = tmpReg[ order( tmpReg$mean ) , ]
  }
  
  tmpReg$predictor <- factor(tmpReg$predictor, levels = tmpReg$predictor[order(tmpReg$mean)])

  h1 = ggplot(tmpReg, aes(x=factor(predictor), y=mean, colour = dotColor)) + coord_flip() +
    geom_errorbar(aes(ymin=lb, ymax=ub), width=.1) +
    geom_line() +
    geom_point() +
    theme(text = element_text(size=9), legend.position="none", axis.text.y = element_text(colour = tmpReg$labColor)) +
    scale_color_manual(values = plotCols) +  # colors for "no" and "yes" (alphabetical order...)
    xlab("Predictors") +
    ylab("Beta estimates") +
    ggtitle("") +
    theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 12), axis.title = element_text(size=15))
  print(h1)
}
