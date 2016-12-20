"""
The following information allows one to reproduce the results
of the analysis in Python.
NOTE: easyml may evolve which is why a commit ID is provided.
Repository: https://github.com/CCS-Lab/easyml
Commit ID: 8e6736c06921ce8c6cd34761bd5c6bf654bf4a74
URL: https://github.com/CCS-Lab/easyml/tree/8e6736c06921ce8c6cd34761bd5c6bf654bf4a74
"""
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os; os.chdir('./Python')
import pandas as pd

from easyml.glmnet import easy_glmnet


# Set matplotlib settings
plt.style.use('ggplot')

if __name__ == "__main__":
    # Load data
    cocaine_dependence = pd.read_table('../data/cocaine_dependence.txt')

    # Analyze data
    easy_glmnet(cocaine_dependence, 'DIAGNOSIS',
                family='binomial', random_state=12345,
                exclude_variables=['subject'], categorical_variables=['Male'], 
                alpha=1, n_lambda=200, standardize=False, cut_point=0, max_iter=1e6)
