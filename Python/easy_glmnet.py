import matplotlib as mpl
import pandas as pd

# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# The following information allows one to reproduce the results
# of the analysis in Python.
# NOTE: easyml may evolve which is why a commit ID is provided.
# Repository: https://github.com/CCS-Lab/easyml
# Commit ID: b486fa85d9e73b6215e5a931e0684c81662e1c81
# URL: https://github.com/CCS-Lab/easyml/tree/b486fa85d9e73b6215e5a931e0684c81662e1c81
from easyml.factory import easy_glmnet

if __name__ == "__main__":
    # Load data
    cocaine = pd.read_table('../data/cocaine.txt')

    # Analyze data
    easy_glmnet(cocaine, 'DIAGNOSIS',
                family='binomial', exclude_variables=['subject'], categorical_variables=['Male'],
                alpha=1, n_lambda=200, standardize=False, cut_point=0, max_iter=1e6)
