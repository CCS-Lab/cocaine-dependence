import matplotlib as mpl
import pandas as pd

# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from easyml.factory import easy_glmnet  # https://github.com/CCS-Lab/easyml


if __name__ == "__main__":
    # Load data
    cocaine = pd.read_table('../data/cocaine.txt')

    # Analyze data
    easy_glmnet(cocaine, dependent_variable='DIAGNOSIS',
                family='binomial', exclude_variables=['subject', 'AGE'])
