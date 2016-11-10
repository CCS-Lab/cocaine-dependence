library(easyml)

# Load data
data("cocaine", package = "easyml")

# Analyze data
easy_glmnet(data = cocaine, dependent_variable = "DIAGNOSIS", family = "binomial",
            exclude_variables = c("subject", "AGE"))
