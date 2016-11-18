# The following information allows one to reproduce the results
# of the analysis in R.
# NOTE: easyml may evolve which is why a commit ID is provided.
# Repository: https://github.com/CCS-Lab/easyml
# Commit ID:
# URL: https://github.com/CCS-Lab/easyml/tree/
library(easyml)

# Load data
data("cocaine", package = "easyml")

# Analyze data
easy_glmnet(cocaine_dependence, "DIAGNOSIS",
            family = "binomial", exclude_variables = c("subject", "AGE"),
            categorical_variables = c("Male"))
