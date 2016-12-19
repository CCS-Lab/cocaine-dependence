# The following information allows one to reproduce the results
# of the analysis in R.
# NOTE: easyml may evolve which is why a commit ID is provided.
# Repository: https://github.com/CCS-Lab/easyml
# Commit ID: 8e6736c06921ce8c6cd34761bd5c6bf654bf4a74
# URL: https://github.com/CCS-Lab/easyml/commit/8e6736c06921ce8c6cd34761bd5c6bf654bf4a74
library(easyml)

# Load data
data("cocaine_dependence", package = "easyml")

# Analyze data
easy_glmnet(cocaine_dependence, "DIAGNOSIS",
            family = "binomial", exclude_variables = c("subject"),
            categorical_variables = c("Male"), standardize = FALSE, 
            random_state = 12345, alpha = 1, nlambda = 200)
