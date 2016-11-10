library(easyml)

data("cocaine", package = "easyml")
easy_glmnet(data = cocaine, dependent_variable = "DIAGNOSIS", family = "binomial",
            exclude_variables = c("subject", "AGE"))
