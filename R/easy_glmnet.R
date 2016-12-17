# The following information allows one to reproduce the results
# of the analysis in R.
# NOTE: easyml may evolve which is why a commit ID is provided.
# Repository: https://github.com/CCS-Lab/easyml
# Commit ID: 61967d60441e6fa44d0a4ff6729395a918e2cb26
# URL: https://github.com/CCS-Lab/easyml/commit/61967d60441e6fa44d0a4ff6729395a918e2cb26
library(easyml)

# Load data
data("cocaine_dependence", package = "easyml")

# Analyze data
easy_glmnet(cocaine_dependence, "DIAGNOSIS",
            family = "binomial", exclude_variables = c("subject"),
            categorical_variables = c("Male"), standardize = FALSE, 
            alpha = 1, nlambda = 200)
