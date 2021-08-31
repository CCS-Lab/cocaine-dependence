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
                       n_samples = 10, n_divisions = 100, n_iterations = 10
                       random_state = 12345, model_args = list(alpha = 1.0))

results$plot_coefficients
results$plot_model_performance_train
results$plot_model_performance_test
results$plot_roc_single_train_test_split_train
results$plot_roc_single_train_test_split_test
