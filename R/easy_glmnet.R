# Repository: https://github.com/CCS-Lab/easyml
# Commit ID:
library(easyml)

# Load data
data("cocaine", package = "easyml")

# Analyze data
easy_glmnet(cocaine, "DIAGNOSIS", family = "binomial",
            exclude_variables = c("subject"), categorical_variables = "Male",
            alpha = 1, n_lambda = 200, standardize = False, cut_point = 0,
            max_iter = 1e6)
