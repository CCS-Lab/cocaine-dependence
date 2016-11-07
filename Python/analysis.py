import matplotlib as mpl

# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from easyml.datasets import cocaine
from easyml.factory import easy_glmnet


if __name__ == "__main__":
    # Load data
    data = cocaine.load_data()

    # Analyze data
    easy_glmnet(data, dependent_variable='DIAGNOSIS', family='binomial', exclude_variables=['subject', 'AGE'])
