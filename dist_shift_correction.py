# -*- coding: utf-8 -*-
"""Smaller version of distribution_shift_correction.ipynb"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
    log_loss
)

sns.set_style()
sklearn.set_config(enable_metadata_routing=True)

DATA_FOLDER = ""

# Source distribution, q(X, y)
X = pd.read_csv(DATA_FOLDER + "X_train.csv")
y = pd.read_csv(DATA_FOLDER + "y_train.csv").squeeze()

CLASSES = np.unique(y)
N_CLASSES = CLASSES.size

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# Target distribution, p(X, y). First 202 X values have corresponding labels, the other 1818 don't
X_shifted = pd.read_csv(DATA_FOLDER + "X_test_2.csv")
X_shifted_labelled = X_shifted.iloc[:202, :]

y_shifted = pd.read_csv(DATA_FOLDER + "y_test_2_reduced.csv").squeeze()

print("Unique values in y_shifted:", np.unique(y_shifted))
print("Classes missing from y_shifted:", np.setdiff1d(y, y_shifted))

# Log loss, weighted up the number of samples in each class (lower is better)
def weighted_log_loss(y_true, y_pred_proba, sample_weight=None):
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    y_true_ohe = label_binarize(y_true, classes=CLASSES)

    if sample_weight is None:
        # Weight the classes by the inverse frequencies
        class_counts = np.sum(y_true_ohe, axis=0)

        # class_weights = 1/class_counts, weighting classes with no samples as 0
        #   https://stackoverflow.com/a/37977222/21453336
        class_weights = np.divide(
            1.0,
            class_counts,
            out=np.zeros(class_counts.shape, dtype=float),
            where=class_counts!=0
        )

        # Normalise the weights so they sum to 1 (for interpretability)
        class_weights /= np.sum(class_weights)

        # Weight each sample by the class weight of the corresponding true y value
        sample_weight = np.sum(y_true_ohe * class_weights, axis=1)

    # Log probabilities are negative, so the mean is too. Negate so the loss is positive.
    loss = -np.mean(sample_weight * np.sum(y_true_ohe * np.log(y_pred_proba), axis=1))
    return loss

# Negative weighted log loss (higher is better)
neg_wll = make_scorer(
    weighted_log_loss, response_method="predict_proba", greater_is_better=False
).set_score_request(sample_weight=True)

# Pass sample weights to fit and score
# https://scikit-learn.org/stable/metadata_routing.html#usage-examples

scaler = StandardScaler().set_fit_request(sample_weight=False)

# lr_cv = LogisticRegressionCV(
#     # Cs=30,
#     Cs=np.logspace(-5, 2, 30), # type: ignore
#     class_weight="balanced",
#     scoring=neg_wll,
#     cv=3, # 3 folds, using StratifiedKFold
#     n_jobs=-1,
# ).set_fit_request(sample_weight=True)


"""Let the training samples be drawn from a "source" distribution $q(X, y)$ and the samples from the second test set be drawn from a "target" distribution $p(X, y).$ The drop in performance (weighted log-loss) likely indicates a shift in the distribution from which the samples were drawn -- that is, $q(X, y) \ne p(X, y)$. This makes the model's training inapplicable to the target distribution.

# Attempting covariate shift correction
Note that the joint distribution can be decomposed using Baye's rule: $q(X, y) = q(y|X)q(X) = q(X|y)q(y)$.

There are three main types of distribution shift -- cases when $q(X, y) \ne p(X, y)$:
1. Covariate shift: $q(X) \ne p(X)$ but $q(y|X) = p(y|X)$, which implies $q(y|X)q(X) \ne p(y|X)p(X)$
2. Label shift: $q(y) \ne p(y)$ but $q(X|y) = p(X|y)$, which implies $q(X|y)q(y) \ne p(X|y)p(y)$
3. Concept drift: $q(y|X) \ne p(y|X)$ but $q(X) = p(X)$, which implies $q(y|X)q(X) \ne p(y|X)p(X)$

(The case when $q(X|y) \ne p(X|y)$ but $q(Y) = p(X)$ would be "too difficult to study", according to [Chip Huyen](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html#fn:20).)
"""


"""# Label shift correction

We can see from the histograms that the distribution of $y$ changes, so we will attempt to correct for label shift.
"""

# Adapted from https://github.com/flaviovdf/label-shift/tree/master?tab=readme-ov-file,
# which is the code
# - https://arxiv.org/pdf/1802.03916

def calculate_marginal(y, n_classes):
    """Calculate P(y)"""
    mu = np.zeros(shape=(n_classes, 1))
    for i in range(n_classes):
        mu[i] = np.sum(y == i)
    return mu / y.shape[0]

def estimate_labelshift_ratio(y_true_val, y_pred_val, y_pred_shifted, n_classes):
    """Estimate class weights for importance-weighted label shift correction,
    P_te(y)/P_tr(y). This favours classes that are more likely to appear in the
    target distribution.
    """
    labels = np.arange(n_classes)
    C = confusion_matrix(y_true_val, y_pred_val, labels=labels).T
    C = C / y_true_val.shape[0]

    mu_t = calculate_marginal(y_pred_shifted, n_classes)
    lamb = 1.0 / min(y_pred_val.shape[0], y_pred_shifted.shape[0])

    I = np.eye(n_classes)
    wt = np.linalg.solve(np.dot(C.T, C) + lamb * I, np.dot(C.T, mu_t))
    return wt.squeeze()

# lshift_class_weight = estimate_labelshift_ratio(y_val, y_pred_val, y_pred_shifted, N_CLASSES)

"""## Alternate approach - directly modelling $\frac{P_{te}(y)}{P_{tr}(y)}$

Rather than estimating $\frac{P_{te}(y)}{P_{tr}(y)}$ using the confusion matrix via $Cp(y) = \mu (\hat y) \Rightarrow p(y) = C^{-1} \mu (\hat y)$, we will try to directly model $P_{te}(y)$ and $P_{tr}(y)$ by the class proportions in each dataset.

This may be a poor approximation of the true distribution ratio because of the low number of labelled samples available (202, of the 2020 samples in the shifted distribution). However, it doesn't rely on potentially inaccurate predictions of the classifier, and doesn't assume the confusion matrix is invertible.
"""

p_tr = calculate_marginal(y, N_CLASSES).squeeze()
p_te = calculate_marginal(y_shifted, N_CLASSES).squeeze()

lshift_class_weight = p_te / p_tr
lshift_sample_weight = np.sum(label_binarize(y, classes=CLASSES) * lshift_class_weight, axis=1)


"""## Balanced weighting based on the test2 distribution

Recall that in the "standard" weighted log-loss formula, the class weights are defined as $w_{y_i} = \frac{1}{f_{y_i}}$, where $f_{y_i}$ is the frequency of class $y_i$ in the test set. This helps mitigate class imbalance issues. We will try to use the frequencies from the shifted distribution instead, assuming the 202 labelled samples are representative. Another approach could be to use the confusion matrix to model $P_{\text{Te}}(y)$ as seen earlier.
"""

# Weight the classes by the inverse frequencies
shifted_class_counts = np.sum(label_binarize(y_shifted, classes=CLASSES), axis=0)
# Note that we could've instead modelled P(y_Te) using
#   calculate_marginal(y_shifted, N_CLASSES).squeeze()
# which is just f_{y_Te} scaled by a constant factor

# class_weights = 1/class_counts, weighting classes with no samples as 0
#   https://stackoverflow.com/a/37977222/21453336
balanced_shifted_class_weight = np.divide(
    1.0,
    shifted_class_counts,
    out=np.zeros(shifted_class_counts.shape, dtype=float),
    where=shifted_class_counts!=0
)

# Normalise the weights so they sum to 1 (for interpretability)
balanced_shifted_class_weight /= np.sum(balanced_shifted_class_weight)

# Weight each sample by the class weight of the corresponding true y value
balanced_shifted_sample_weight = np.sum(
    label_binarize(y, classes=CLASSES) * balanced_shifted_class_weight,
    axis=1
)

"""## Chaining the weights to correct the class imbalance of the target distribution
Try using the importance-weighted, label-weighted log loss. This is in the hopes of
minimising the expected weighted log loss under the target distribution.

Note that this is identical to simply weighting by class weights of the train set,
as (P_te(y) / P_tr(y)) * (1 / P_te(y)) = 1 / P_tr(y).
"""
wt = lshift_sample_weight * balanced_shifted_sample_weight

lr_label_shift = LogisticRegressionCV(
    Cs=np.logspace(-5, 2, 10), # type: ignore
    scoring=neg_wll,
    cv=3,
    n_jobs=-1,
).set_fit_request(sample_weight=True)

pipe = Pipeline(steps=[
    ("scaler", scaler),
    ("classifier", lr_label_shift),
])

pipe.fit(X, y, sample_weight=wt)
print("Best C:", np.median(pipe.named_steps['classifier'].C_))

# Predict on the target (shifted) distribution
pred = pipe.predict(X_shifted_labelled)
proba = pipe.predict_proba(X_shifted_labelled)

print("Weighted log-loss on shifted dataset:", weighted_log_loss(y_shifted, proba))
plot_metrics(y_shifted, pred, "LogisticRegressionCV weighted by P_te(y)/P_tr(y) * 1/P_te(y), Distribution Shifted Data")
plt.savefig("dist_shift_correction_submission.png")
