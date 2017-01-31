# The following information allows one to reproduce the results
# of the analysis in R.
# NOTE: easyml may evolve which is why a commit ID is provided.
# Repository: https://github.com/CCS-Lab/easyml
# Commit ID: dac38d697ab5adf2faaaab68d6aeb5bd0d729a94
# URL: https://github.com/CCS-Lab/easyml/commit/dac38d697ab5adf2faaaab68d6aeb5bd0d729a94
library(easyml)

# Load data
data("cocaine_dependence", package = "easyml")

# Analyze data
results <- easy_glmnet(cocaine_dependence, "diagnosis",
                       family = "binomial", preprocess = preprocess_scale, 
                       exclude_variables = c("subject"), categorical_variables = c("male"), 
                       random_state = 12345, alpha = 1, nlambda = 200)

results$plot_coefficients_processed
results$plot_predictions_train_mean
results$plot_predictions_test_mean
results$plot_metrics_train_mean
results$plot_metrics_train_mean
