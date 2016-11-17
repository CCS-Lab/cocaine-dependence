from glmnet import LogitNet
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sample import sample_equal_proportion


# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Constants
EXCLUDE_AGE = False
TRAIN_SIZE = 0.667
MAX_ITER = 1e6
ALPHA = 1
N_LAMBDA = 200
N_FOLDS = 5
N_DIVISIONS = 1000
N_ITERATIONS = 10
CUT_POINT = 0  # use 0 for minimum, 1 for within 1 SE
SURVIVAL_RATE_CUTOFF = 0.05
SHOW = False
SAVE = True

# Load data
data = pd.read_table('../data/cocaine.txt')

# Drop subjects column
data = data.drop('subject', axis=1)

# Possibly exclude age
if EXCLUDE_AGE:
    data = data.drop('AGE', axis=1)

# Handle dependent variables
y = data['DIAGNOSIS'].values
data = data.drop('DIAGNOSIS', axis=1)

# Handle categorical variable
male = np.array([data['Male'].values]).T
data = data.drop('Male', axis=1)
X_raw = data.values

# Handle numeric variables
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X_raw)

# Combine categorical variables and continuous variables
X = np.concatenate([male, X_std], axis=1)

##############################################################################
# Replicating figure 1 - Done!
##############################################################################
# Create temporary containers
coefs = []

# Loop over number of iterations
for i in tqdm(range(N_ITERATIONS)):
    # Fit LogisticNet with the training set
    lr = LogitNet(alpha=ALPHA, n_lambda=N_LAMBDA, standardize=False, cut_point=CUT_POINT, max_iter=MAX_ITER)
    lr = lr.fit(X, y)

    # Extract and save coefficients
    coefs.append(list(lr.coef_[0]))

coefs = np.array(coefs)
survived = 1 * (abs(coefs) > 0)
survival_rate = np.sum(survived, axis=0) / float(N_ITERATIONS)
mask = 1 * (survival_rate > SURVIVAL_RATE_CUTOFF)
coefs_updated = coefs * mask
variable_names = ['Male'] + list(data.columns)
coefs_q025 = np.percentile(coefs_updated, q=2.5, axis=0)
coefs_mean = np.mean(coefs_updated, axis=0)
coefs_q975 = np.percentile(coefs_updated, q=97.5, axis=0)
betas = pd.DataFrame({'mean': coefs_mean})
betas['lb'] = coefs_q025
betas['ub'] = coefs_q975
betas['survival'] = mask
betas['predictor'] = variable_names
betas['sig'] = betas['survival']
betas['dotColor1'] = 1 * (betas['mean'] != 0)
betas['dotColor2'] = (1 * np.logical_and(betas['dotColor1'] > 0, betas['sig'] > 0)) + 1
betas['dotColor'] = betas['dotColor1'] * betas['dotColor2']
betas.to_csv('./betas.csv', index=False)

##############################################################################
# Replicating figure 2 - Done!
##############################################################################
# Split data
mask = sample_equal_proportion(y, proportion=TRAIN_SIZE, random_state=43210)
y_train = y[mask]
y_test = y[np.logical_not(mask)]
X_train = X[mask, :]
X_test = X[np.logical_not(mask), :]

# Create temporary containers
all_y_train_scores = []
all_y_test_scores = []

# Loop over number of iterations
for i in tqdm(range(N_ITERATIONS)):
    # Fit LogisticNet with the training set
    lr = LogitNet(alpha=ALPHA, n_lambda=N_LAMBDA, standardize=False, n_folds=N_FOLDS, max_iter=MAX_ITER, random_state=i)
    lr = lr.fit(X_train, y_train)

    # Identify and save the best lambda
    lamb = lr.lambda_max_

    # Generate scores for training and test sets
    y_train_scores = lr.predict_proba(X_train, lamb=lamb)[:, 1]
    y_test_scores = lr.predict_proba(X_test, lamb=lamb)[:, 1]

    # Save AUCs
    all_y_train_scores.append(y_train_scores)
    all_y_test_scores.append(y_test_scores)

# Generate scores for training and test sets
all_y_train_scores = np.array(all_y_train_scores)
y_train_scores_mean = np.mean(all_y_train_scores, axis=0)
all_y_test_scores = np.array(all_y_test_scores)
y_test_scores_mean = np.mean(all_y_test_scores, axis=0)

# Compute ROC curve and ROC area for each class
n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_train, y_train_scores_mean)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute train ROC curve
plt.figure()
plt.plot(fpr[1], tpr[1], color='black',
         lw=2, label='AUC = %.3f' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve (Training Set)')
plt.legend(loc="lower right")
if SHOW:
    plt.show()
if SAVE:
    plt.savefig('./train_roc_curve.png')

# Compute ROC curve and ROC area for each class
n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test, y_test_scores_mean)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute test ROC curve
plt.figure()
plt.plot(fpr[1], tpr[1], color='black',
         lw=2, label='AUC = %.3f' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve (Test Set)')
plt.legend(loc="lower right")
if SHOW:
    plt.show()
if SAVE:
    plt.savefig('./test_roc_curve.png')

##############################################################################
# Replicating figure 4 - Done!
##############################################################################
# Create temporary containers
all_train_aucs = []
all_test_aucs = []

# Loop over number of divisions
for i in tqdm(range(N_DIVISIONS)):
    # Split data
    mask = sample_equal_proportion(y, proportion=TRAIN_SIZE, random_state=i)
    y_train = y[mask]
    y_test = y[np.logical_not(mask)]
    X_train = X[mask, :]
    X_test = X[np.logical_not(mask), :]

    # Create temporary containers
    train_aucs = []
    test_aucs = []

    # Loop over number of iterations
    for j in range(N_ITERATIONS):
        # Fit LogisticNet with the training set
        lr = LogitNet(alpha=ALPHA, n_lambda=N_LAMBDA, standardize=False, n_folds=N_FOLDS, max_iter=MAX_ITER, random_state=j)
        lr = lr.fit(X_train, y_train)

        # Identify and save the best lambda
        lamb = lr.lambda_max_

        # Generate scores for training and test sets
        y_train_scores = lr.predict_proba(X_train, lamb=lamb)[:, 1]
        y_test_scores = lr.predict_proba(X_test, lamb=lamb)[:, 1]

        # Calculate AUC on training and test sets
        train_auc = metrics.roc_auc_score(y_train, y_train_scores)
        test_auc = metrics.roc_auc_score(y_test, y_test_scores)

        # Save AUCs
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

    # Process loop and save in temporary containers
    all_train_aucs.append(np.mean(train_aucs))
    all_test_aucs.append(np.mean(test_aucs))

all_train_aucs = np.array(all_train_aucs)
all_train_auc_mean = np.mean(all_train_aucs)
all_test_aucs = np.array(all_test_aucs)
all_test_auc_mean = np.mean(all_test_aucs)
bins = np.arange(0, 1, 0.02)

plt.figure()
plt.hist(all_train_aucs, bins=bins, color='white', edgecolor='black')
plt.axvline(x=all_train_auc_mean, color='black', linestyle='--')
plt.annotate('Mean AUC = %.3f' % all_train_auc_mean, xy=(150, 200), xycoords='figure pixels', size=28)
plt.xlim([0.0, 1.0])
plt.xlabel('AUC')
plt.ylabel('Frequency')
plt.title('Distribution of AUCs (Training Set)')
if SHOW:
    plt.show()
if SAVE:
    plt.savefig('./train_auc_distribution.png')

plt.figure()
plt.hist(all_test_aucs, bins=bins, color='white', edgecolor='black')
plt.axvline(x=all_test_auc_mean, color='black', linestyle='--')
plt.annotate('Mean AUC = %.3f' % all_test_auc_mean, xy=(150, 200), xycoords='figure pixels', size=28)
plt.xlim([0.0, 1.0])
plt.xlabel('AUC')
plt.ylabel('Frequency')
plt.title('Distribution of AUCs (Test Set)')
if SHOW:
    plt.show()
if SAVE:
    plt.savefig('./test_auc_distribution.png')
