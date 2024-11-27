# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
import scipy.optimize
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform



# %%
# load data
df = load_transform()
print(df.head(10))



# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
# We divide by Exposure to normalise the claim amounts relative to the level exposure. 
# This, in turn, provides a standardised measure of risk per unit measure of exposure.
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]

# TODO: use your create_sample_split function here
df = create_sample_split(df,id_column="IDpol")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)


# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here
        ("num", Pipeline([
            ("scaler", StandardScaler()),
            ("spline", SplineTransformer(knots="quantile", n_knots=4, degree=3, include_bias=False))
        ]), numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
    # TODO: Define pipeline steps here
    [
        ("preprocessor", preprocessor),
        ("glm", GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True))
    ]
)

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, glm__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)



# %% [markdown]
#  ## How does the deviance on the train and test set change?
# 
#  How could we check whether we are overfitting?
# 
#  The deviance on the training and testing sets is more closely aligned when **BonusMalus** and **VehPower** are modeled as simple linear terms. However, in both cases, the deviance values for the training and testing sets are sufficiently close to suggest that overfitting is not a significant concern. Notably, when **BonusMalus** and **VehPower** are modeled using scalers and splines, the predicted values show better alignment with the observed data, indicating that this approach might produce a more accurate model than assuming these variables follow a linear relationship.
# 
# 
# 
#  However, caution must be exercised to avoid overfitting. To ensure the robustness of the model, several strategies can be employed:
# 
#  1. **Cross-Validation**: Evaluate the model's performance on multiple subsets of the data to confirm its generalizability.
# 
#  2. **Learning Curves**: Plot learning curves to visualize how the training and validation performance evolve as the training set size increases.
# 
#  3. **Regularization**: Incorporate penalties or constraints into the model (e.g., L1 or L2 regularization) to prevent overfitting and improve generalization.
# 
# 
# 
#  These steps will help assess and improve the model's ability to perform well on unseen data while leveraging the improved predictive power of non-linear modeling techniques.
# 
# 

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.
# Define the modelling pipeline
model_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("lgbm", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5))
    ]
)

model_pipeline.fit(X_train_t, y_train_t, lgbm__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)



# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
param_grid = {
    "lgbm__learning_rate": [0.01, 0.05, 0.1],
    "lgbm__n_estimators": [100, 200, 300]
}

cv = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_poisson_deviance",
    verbose=1,
    n_jobs=-1
)
cv.fit(X_train_t, y_train_t, lgbm__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)


# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()


# %% [markdown]
# # Optional Exercise
# 
# 
# 
# 
# 
#  **Intro**
# 
# 
# 
#  Copy the GLM tutorial code for the Poisson and Gamma to model the frequency and severity of the claims.
# 
#  Then combine the two models to predict the total claim amount per policy.
# 
#  I use the same approach as in the GLM tutorial to model the frequency and severity of the claims. But with LGBM model.
# 
#  Then combine the two models to predict the total claim amount per policy.
# 
# 
# 
# 
# 
#  GLM tutorial shows why and how to use Poisson, Gamma, and Tweedie GLMs on an insurance claims dataset using `glum`. It was inspired by, and closely mirrors, two other GLM tutorials that used this dataset:
# 
# 
# 
#  1. An sklearn-learn tutorial, [Tweedie regression on insurance claims](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html#pure-premium-modeling-via-a-product-model-vs-single-tweedieregressor), which was created for this (partially merged) [sklearn PR](https://github.com/scikit-learn/scikit-learn/pull/9405) that we based glum on
# 
#  2. An R tutorial, [Case Study: French Motor Third-Party Liability Claims](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3164764) with [R code](https://github.com/JSchelldorfer/ActuarialDataScience/tree/master/1%20-%20French%20Motor%20Third-Party%20Liability%20Claims).
# 
# 
# 
# 
# 
#  **Background**
# 
# 
# 
#  Insurance claims are requests made by a policy holder to an insurance company for compensation in the event of a covered loss. When modeling these claims, the goal is often to estimate, per policy, the total claim amount per exposure unit. (i.e. number of claims $\times$ average amount per claim per year). This amount is also referred to as the pure premium.
# 
# 
# 
#  Two approaches for modeling this value are:
# 
# 
# 
#  1. Modeling the total claim amount per exposure directly
# 
#  2. Modeling number of claims and claim amount separately with a frequency and a severity model

# %% [markdown]
#  ## 2. Frequency GLM - Poisson distribution<a class="anchor"></a>
# 
#  [back to Table of Contents](#Table-of-Contents)
# 
# 
# 
#  We start with the first part of our two part GLM - modeling the frequency of claims using a Poisson regression. Below, we give some background on why the Poisson family makes the most sense in this context.
# 
# 
# 
#  ### 2.1 Why Poisson distributions?
# 
#  Poisson distributions are typically used to model the number of events occurring in a fixed period of time when the events occur independently at a constant rate. In our case, we can think of motor insurance claims as the events, and a unit of exposure (i.e. a year) as the fixed period of time.
# 
# 
# 
#  To get more technical:
# 
# 
# 
#  We define:
# 
# 
# 
#  - $z$: number of claims
# 
#  - $w$: exposure (time in years under risk)
# 
#  - $y = \frac{z}{w}$: claim frequency per year
# 
#  - $X$: feature matrix
# 
# 
# 
# 
# 
#  The number of claims $z$ is an integer, $z \in [0, 1, 2, 3, \ldots]$. Theoretically, a policy could have an arbitrarily large number of claims&mdash;very unlikely but possible. The simplest distribution for this range is a Poisson distribution $z \sim Poisson$. However, instead of $z$, we will model the frequency $y$. Nonetheless, this  is still (scaled) Poisson distributed with variance inverse proportional to $w$, cf. [wikipedia:Reproductive_EDM](https://en.wikipedia.org/wiki/Exponential_dispersion_model#Reproductive).
# 
# 
# 
#  To verify our assumptions, we start by plotting the observed frequencies and a fitted Poisson distribution (Poisson regression with intercept only).

# %%
# plt.subplots(figsize=(10, 7))
df_plot = (
    df.loc[:, ['ClaimNb', 'Exposure']].groupby('ClaimNb').sum()
    .assign(Frequency_Observed = lambda x: x.Exposure / df['Exposure'].sum())
)
mean = df['ClaimNb'].sum() / df['Exposure'].sum()

x = range(5)
plt.scatter(x, df_plot['Frequency_Observed'].values, color="blue", alpha=0.85, s=60, label='observed')
plt.scatter(x, scipy.stats.poisson.pmf(x, mean), color="orange", alpha=0.55, s=60, label="poisson fit")
plt.xticks(x)
plt.legend()
plt.title("Frequency");


# %% [markdown]
#  This is a strong confirmation for the use of a Poisson when fitting!
# 
# 
# 
# 
# 
#  ### 2.2 Train and test frequency GLM
# 
# 
# 
#  Now, we start fitting our model. We use claims frequency = claim number/exposure as our outcome variable. We then divide the dataset into training set and test set with a 9:1 random split.
# 
# 
# 
#  Also, notice that we do not one hot encode our columns. Rather, we take advantage of `glum`'s integration with `tabmat`, which allows us to pass in categorical columns directly! `tabmat` will handle the encoding for us and even includes a handful of helpful matrix operation optimizations. We use the `Categorizer` from [dask_ml](https://ml.dask.org/modules/generated/dask_ml.preprocessing.Categorizer.html) to set our categorical columns as categorical dtypes and to ensure that the categories align in fitting and predicting.

# %%
z = df['ClaimNb'].values
weight = df['Exposure'].values
y = z / weight # claims frequency

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]
predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_p = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_p = glm_categorizer.transform(df[predictors].iloc[test])
y_train_p, y_test_p = y[train], y[test]
w_train_p, w_test_p = weight[train], weight[test]
z_train_p, z_test_p = z[train], z[test]


# %% [markdown]
#  Now, we define our GLM using the `GeneralizedLinearRegressor` class from `glum`.
# 
# 
# 
#  - `family='poisson'`: creates a Poisson regressor
# 
#  - `alpha_search=True`: tells the GLM to search along the regularization path for the best alpha
# 
#  - `l1_ratio = 1` tells the GLM to only use l1 penalty (not l2). `l1_ratio` is the elastic net mixing parameter. For ``l1_ratio = 0``, the penalty is an L2 penalty. ``For l1_ratio = 1``, it is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.
# 
# 
# 
#  See the `GeneralizedLinearRegressor` class [API documentation](https://glum.readthedocs.io/en/latest/api/modules.html) for more details.
# 
# 
# 
#  *Note*: `glum` also supported a cross validation model GeneralizedLinearRegressorCV. However, because cross validation requires fitting many models, it is much slower and we don’t demonstrate it in this tutorial.

# %%
f_glm1 = GeneralizedLinearRegressor(family='poisson', alpha_search=True, l1_ratio=1, fit_intercept=True)

f_glm1.fit(
    X_train_p,
    y_train_p,
    sample_weight=w_train_p
);

pd.DataFrame({'coefficient': np.concatenate(([f_glm1.intercept_], f_glm1.coef_))},
             index=['intercept'] + f_glm1.feature_names_).T


# %% [markdown]
#  To measure our model's test and train performance, we use the deviance function for the Poisson family. We can get the total deviance function directly from `glum`'s distribution classes and divide it by the sum of our sample weight.
# 
# 
# 
#  *Note*: a Poisson distribution is equivalent to a Tweedie distribution with power = 1.

# %%
PoissonDist = TweedieDistribution(1)
print('training loss f_glm1: {}'.format(
    PoissonDist.deviance(y_train_p, f_glm1.predict(X_train_p), sample_weight=w_train_p)/np.sum(w_train_p)
))

print('test loss f_glm1: {}'.format(
      PoissonDist.deviance(y_test_p, f_glm1.predict(X_test_p), sample_weight=w_test_p)/np.sum(w_test_p)))


# %% [markdown]
#  A GLM with canonical link function (Normal - identity, Poisson - log, Gamma - 1/x, Binomial - logit) with an intercept term has the so called **balance property**. Neglecting small deviations due to an imperfect fit, on the training sample the results satisfy the equality:
# 
#  $$\sum_{i \in training} w_i y_i = \sum_{i \in training} w_i \hat{\mu}_i$$
# 
#  As expected, this property holds in our real data:

# %%
# balance property of GLM with canonical link, like log-link for Poisson:
z_train_p.sum(), (f_glm1.predict(X_train_p) * w_train_p).sum()


# %% [markdown]
#  ## 3. Severity GLM - Gamma distribution<a class="anchor"></a>
# 
#  [back to Table of Contents](#Table-of-Contents)
# 
# 
# 
#  Now, we fit a GLM for the severity with the same features as the frequency model.
# 
#  The severity $y$ is the average claim size.
# 
#  We define:
# 
# 
# 
#  - $z$: total claim amount, single claims cut at 100,000
# 
#  - $w$: number of claims (with positive claim amount!)
# 
#  - $y = \frac{z}{w}$: severity
# 
# 
# 
#  ### 3.1 Why Gamma distributions
# 
#  The severity $y$ is a positive, real number, $y \in (0, \infty)$. Theoretically, especially for liability claims, one could have arbitrary large numbers&mdash;very unlikely but possible. A very simple distribution for this range is an Exponential distribution, or its generalization, a Gamma distribution $y \sim Gamma$. In the insurance industry, it is well known that the severity might be skewed by a few very large losses. It's common to model these tail losses separately so here we cut out claims larger than 100,000 to focus on modeling small and moderate claims.

# %%
df_plot = (
    df.loc[:, ['ClaimAmountCut', 'ClaimNb']]
    .query('ClaimNb > 0')
    .assign(Severity_Observed = lambda x: x['ClaimAmountCut'] / df['ClaimNb'])
)

df_plot['Severity_Observed'].plot.hist(bins=400, density=True, label='Observed', )

x = np.linspace(0, 1e5, num=400)
plt.plot(x,
         scipy.stats.gamma.pdf(x, *scipy.stats.gamma.fit(df_plot['Severity_Observed'], floc=0)),
         'r-', label='fitted Gamma')
plt.legend()
plt.title("Severity");
plt.xlim(left=0, right = 1e4);
#plt.xticks(x);


# %%
# Check mean-variance relationship for Gamma: Var[Y] = E[Y]^2 / Exposure
# Estimate Var[Y] and E[Y]
# Plot estimates Var[Y] vs E[Y]^s/Exposure
# Note: We group by VehPower and BonusMalus in order to have different E[Y].

def my_agg(x):
    """See https://stackoverflow.com/q/44635626"""
    x_sev = x['Sev']
    x_cnb = x['ClaimNb']
    n = x_sev.shape[0]
    names = {
        'Sev_mean': np.average(x_sev, weights=x_cnb),
        'Sev_var': 1/(n-1) * np.sum((x_cnb/np.sum(x_cnb)) * (x_sev-np.average(x_sev, weights=x_cnb))**2),
        'ClaimNb_sum': x_cnb.sum()
    }
    return pd.Series(names, index=['Sev_mean', 'Sev_var', 'ClaimNb_sum'])

for col in ['VehPower', 'BonusMalus']:
    claims = df.groupby(col)['ClaimNb'].sum()
    df_plot = (df.loc[df[col].isin(claims[claims >= 4].index), :]
               .query('ClaimNb > 0')
               .assign(Sev = lambda x: x['ClaimAmountCut']/x['ClaimNb'])
               .groupby(col)
               .apply(my_agg)
              )

    plt.plot(df_plot['Sev_mean'], df_plot['Sev_var'] * df_plot['ClaimNb_sum'], '.',
             markersize=12, label='observed')

    # fit: mean**p/claims
    p = scipy.optimize.curve_fit(lambda x, p: np.power(x, p),
                           df_plot['Sev_mean'].values,
                           df_plot['Sev_var'] * df_plot['ClaimNb_sum'],
                           p0 = [2])[0][0]
    df_fit = pd.DataFrame({'x': df_plot['Sev_mean'],
                           'y': np.power(df_plot['Sev_mean'], p)})
    df_fit = df_fit.sort_values('x')

    plt.plot(df_fit.x, df_fit.y,
             'k--', label='fit: Mean**{}'.format(p))
    plt.xlabel('Mean of Severity ')
    plt.ylabel('Variance of Severity * ClaimNb')
    plt.legend()
    plt.title('Man-Variance of Claim Severity by {}'.format(col))
    plt.show()


# %% [markdown]
#  Great! A Gamma distribution seems to be an empirically reasonable assumption for this dataset.
# 
# 
# 
# 
# 
#  *Hint*: If Y were normal distributed, one should see a horizontal line, because $Var[Y] = constant/Exposure$
# 
#         and the fit should give $p \approx 0$.

# %% [markdown]
#  ### 3.2 Severity GLM with train and test data
# 
#  We fit a GLM for the severity with the same features as the frequency model. We use the same categorizer as before.
# 
# 
# 
#  *Note*:
# 
# 
# 
#  - We filter out ClaimAmount == 0. The severity problem is to model claim amounts conditional on a claim having already been submitted. It seems reasonable to treat a claim of zero as equivalent to no claim at all. Additionally, zero is not included in the open interval $(0, \infty)$ support of the Gamma distribution.
# 
#  - We use ClaimNb as sample weights.
# 
#  - We use the same split in train and test data such that we can predict the final claim amount on the test set as the product of our Poisson claim number and Gamma claim severity GLMs.

# %%
idx = df['ClaimAmountCut'].values > 0

z = df['ClaimAmountCut'].values
weight = df['ClaimNb'].values
# y = claims severity
y = np.zeros_like(z)  # zeros will never be used
y[idx] = z[idx] / weight[idx]

# we also need to represent train and test as boolean indices
itrain = np.zeros(y.shape, dtype='bool')
itest = np.zeros(y.shape, dtype='bool')
itrain[train] = True
itest[test] = True
# simplify life
itrain = idx & itrain
itest = idx & itest

X_train_g = glm_categorizer.fit_transform(df[predictors].iloc[itrain])
X_test_g = glm_categorizer.transform(df[predictors].iloc[itest])
y_train_g, y_test_g = y[itrain], y[itest]
w_train_g, w_test_g = weight[itrain], weight[itest]
z_train_g, z_test_g = z[itrain], z[itest]


# %% [markdown]
#  We fit our model with the same parameters before, but of course, this time we use `family=gamma`.

# %%
s_glm1 = GeneralizedLinearRegressor(family='gamma', alpha_search=True, l1_ratio=1, fit_intercept=True)
s_glm1.fit(X_train_g, y_train_g, sample_weight=weight[itrain])

pd.DataFrame({'coefficient': np.concatenate(([s_glm1.intercept_], s_glm1.coef_))},
             index=['intercept'] + s_glm1.feature_names_).T


# %% [markdown]
#  Again, we measure performance with the deviance of the distribution. We also compare against the simple arithmetic mean and include the mean absolute error to help understand the actual scale of our results.
# 
# 
# 
#  *Note*: a Gamma distribution is equivalent to a Tweedie distribution with power = 2.

# %%
GammaDist = TweedieDistribution(2)
print('training loss (deviance) s_glm1:     {}'.format(
    GammaDist.deviance(
        y_train_g, s_glm1.predict(X_train_g), sample_weight=w_train_g
    )/np.sum(w_train_g)
))
print('training mean absolute error s_glm1: {}'.format(
    mean_absolute_error(y_train_g, s_glm1.predict(X_train_g))
))

print('\ntesting loss s_glm1 (deviance):      {}'.format(
    GammaDist.deviance(
        y_test_g, s_glm1.predict(X_test_g), sample_weight=w_test_g
    )/np.sum(w_test_g)
))
print('testing mean absolute error s_glm1:  {}'.format(
    mean_absolute_error(y_test_g, s_glm1.predict(X_test_g))
))

print('\ntesting loss Mean (deviance):        {}'.format(
    GammaDist.deviance(
        y_test_g, np.average(z_train_g, weights=w_train_g)*np.ones_like(z_test_g), sample_weight=w_test_g
    )/np.sum(w_test_g)
))
print('testing mean absolute error Mean:    {}'.format(
    mean_absolute_error(y_test_g, np.average(z_train_g, weights=w_train_g)*np.ones_like(z_test_g))
))


# %% [markdown]
#  Even though the deviance improvement seems small, the improvement in mean absolute error is not! (In the insurance world, this will make a significant difference when aggregated over all claims).

# %% [markdown]
#  ### 3.3 Combined frequency and severity results
# 
# 
# 
#  We put together the prediction of frequency and severity to get the predictions of the total claim amount per policy.

# %%
#Put together freq * sev together
print("Total claim amount on train set, observed = {}, predicted = {}".
     format(df['ClaimAmountCut'].values[train].sum(),
            np.sum(df['Exposure'].values[train] * f_glm1.predict(X_train_p) * s_glm1.predict(X_train_p)))
     )

print("Total claim amount on test set, observed = {}, predicted = {}".
     format(df['ClaimAmountCut'].values[test].sum(),
            np.sum(df['Exposure'].values[test] * f_glm1.predict(X_test_p) * s_glm1.predict(X_test_p)))
     )


# %% [markdown]
# ### 4.1 Frequency model using LGBM with GridsearchCV improvement

# %%
LgbmDist = TweedieDistribution(power=1.5)

model_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("lgbm", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5))
    ]
)

param_grid = {
    "lgbm__learning_rate": [0.01, 0.05, 0.1],
    "lgbm__n_estimators": [100, 200, 300]
}

cv_freq = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_poisson_deviance",
    verbose=1,
    n_jobs=-1
)

cv_freq.fit(X_train_p, y_train_p, lgbm__sample_weight=w_train_p)
freq_train_pred = cv_freq.best_estimator_.predict(X_train_p)
freq_test_pred = cv_freq.best_estimator_.predict(X_test_p)

print('LGBM frequency model training loss: {}'.format(
    LgbmDist.deviance(y_train_p, freq_train_pred, sample_weight=w_train_p) / np.sum(w_train_p)
))
print('LGBM frequency model test loss: {}'.format(
    LgbmDist.deviance(y_test_p, freq_test_pred, sample_weight=w_test_p) / np.sum(w_test_p)
))


# %% [markdown]
# ### 3.2 Severity model with LGBM and GridSearchCV improvement

# %%
param_grid = {
    "lgbm__learning_rate": [0.01, 0.05, 0.1],
    "lgbm__n_estimators": [100, 200, 300]
}

cv_sev = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_poisson_deviance",
    verbose=1,
    n_jobs=-1
)
cv_sev.fit(X_train_g, y_train_g, lgbm__sample_weight=w_train_g)

sev_train_pred_cv = cv_sev.best_estimator_.predict(X_train_g)
sev_test_pred_cv = cv_sev.best_estimator_.predict(X_test_g)

print('LGBM severity model training loss: {}'.format(
    LgbmDist.deviance(y_train_g, sev_train_pred_cv, sample_weight=w_train_g) / np.sum(w_train_g)
))
print('LGBM severity model test loss: {}'.format(
    LgbmDist.deviance(y_test_g, sev_test_pred_cv, sample_weight=w_test_g) / np.sum(w_test_g)
))

# %%
print(z_train_p.sum(), (freq_train_pred * w_train_p).sum())
print(z_train_g.sum(), (sev_train_pred_cv  * w_train_g).sum())
# print(z_train_g.sum(), (sev_train_pred_cv  * w_train_g).sum())

# %%
# Put together freq * sev together for LGBM models
sev_train_pred_cv_p = cv_sev.best_estimator_.predict(X_train_p)
sev_test_pred_cv_p = cv_sev.best_estimator_.predict(X_test_p)

print("Total claim amount on train set, observed = {}, predicted = {}".format(
    df['ClaimAmountCut'].values[train].sum(),
    np.sum(df['Exposure'].values[train] * freq_train_pred * sev_train_pred_cv_p)
))

print("Total claim amount on test set, observed = {}, predicted = {}".format(
    df['ClaimAmountCut'].values[test].sum(),
    np.sum(df['Exposure'].values[test] * freq_test_pred * sev_test_pred_cv_p)
))


# %%
# PS4 EX2
import lightgbm as lgb
# Re-fit the best constrained LGBMRegressor
best_lgbm = cv_freq.best_estimator_  # Best model from cross-validation
eval_results = {}  # Dictionary to store evaluation metrics

# Fit the model with evaluation sets
best_lgbm.named_steps['lgbm'].fit(
    X_train_p,
    y_train_p,
    sample_weight=w_train_p,
    eval_set=[(X_train_p, y_train_p), (X_test_p, y_test_p)],
    eval_names=["train", "test"],
    eval_metric="poisson",  # Scoring metric
    callbacks=[lgb.record_evaluation(eval_results)]  # Record results for plotting
)

# Plot the learning curve
lgb.plot_metric(eval_results, metric="poisson")
plt.title("Learning Curve for LGBMRegressor")
plt.ylabel("Poisson Deviance")
plt.xlabel("Boosting Iterations")
plt.show()

# The model is likely close to its optimal state. Unless the gap between train and test losses increases significantly (indicating overfitting), the current tuning is acceptable. However, additional tuning could be explored if further gains in test performance are needed.
# %%
# PS4 EX3
from ps3.evaluation import evaluate_predictions

# Predictions for unconstrained model
y_pred_unconstrained = model_pipeline.fit(X_train_p, y_train_p).predict(X_test_p)

# Evaluate metrics
metrics_constrained = evaluate_predictions(y_test_p, freq_test_pred, sample_weight=w_test_p)
metrics_unconstrained = evaluate_predictions(y_test_p, y_pred_unconstrained, sample_weight=w_test_p)

# Print results
print("Constrained LGBM Model Metrics:\n", metrics_constrained)
print("\nUnconstrained LGBM Model Metrics:\n", metrics_unconstrained)
# %%
