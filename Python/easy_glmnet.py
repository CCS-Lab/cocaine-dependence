"""
The following information allows one to reproduce the results
of the analysis in Python.
NOTE: easyml may evolve which is why a commit ID is provided.
Repository: https://github.com/CCS-Lab/easyml
Commit ID: 4f21118ae2b531a2fa3993826d8084455c58398f
URL: https://github.com/CCS-Lab/easyml/tree/4f21118ae2b531a2fa3993826d8084455c58398f
"""
import matplotlib.pyplot as plt
import os
import pandas as pd

from easyml.glmnet import EasyGlmnet


# Set matplotlib settings
plt.style.use('ggplot')

# Load data
directory = '../data/'
cocaine_dependence = pd.read_table(os.path.join(directory, 'cocaine_dependence.txt'))

# Analyze data
output = EasyGlmnet(cocaine_dependence, 'DIAGNOSIS',
                    family='binomial',
                    exclude_variables=['subject'],
                    categorical_variables=['Male'],
                    random_state=12345, progress_bar=True, n_core=1,
                    model_args={'alpha': 1, 'n_lambda': 200})
