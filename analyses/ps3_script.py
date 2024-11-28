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
import lightgbm as lgb

from ps3.data import create_sample_split, load_transform
from ps3.evaluation import evaluate_predictions

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
# Let's add splines for BonusMalus and Density and use a Pipeline.
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
        ("num", Pipeline([
            ("scaler", StandardScaler()),
            ("spline", SplineTransformer(knots="quantile", n_knots=4, degree=3, include_bias=False))
        ]), numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)
preprocessor.set_output(transform="pandas")
model_pipeline = Pipeline(
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




# %%
# Let's use a GBM instead as an estimator.
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
# Let's tune the LGBM to reduce overfitting.
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
# ## Exercise 4 Starts

# %%
# Create a plot of the average claims per BonusMalus group weighted by exposure.
average_claims = df.groupby("BonusMalus").apply(lambda x: np.average(x["PurePremium"], weights=x["Exposure"]))
plt.figure(figsize=(12, 6))
average_claims.plot(kind='bar')
plt.xlabel('BonusMalus')
plt.ylabel('Average Claims')
plt.title('Average Claims per BonusMalus Group Weighted by Exposure')
plt.xticks(rotation=45)
plt.locator_params(axis='x', nbins=12)
plt.show()

# %% [markdown]
# ### What will/could happen if we do not include a monotonicity constraint?
# We can see from the above plot that we do not have exposure in every BonusMalus group. It is likely that there might be edge cases in which the monotonicity breaks. Serveral issues could arise if we do not include a monotonicity constraint in the model.
# 
# **Non-intuitive Results:** The model might produce results that do not align with domain knowledge or intuition. For example, the price decreases for customers with lower bonus malus score. 
# 
# **Overfitting**: Without constraints, the model might overfit the training data by capturing noise, leading to poor generalization on new data.
# 
# **Interpretability**: Monotonicity constraints can make the model more interpretable, as they enforce a consistent relationship between input and output variables.
# 
# In summary, including a monotonicity constraint helps ensure that the model behaves in a predictable and interpretable manner, aligning with domain knowledge and improving generalization.

# %%
# Create a new model pipeline or estimator called constrained_lgbm. Introduce an increasing monotonicity constrained for BonusMalus. 
# Note: We have to provide a list of the same length as our features with 0s everywhere except for BonusMalus where we put a 1. 
# See: https://lightgbm.readthedocs.io/en/latest/Parameters.html

monotone_constraints = [0] * len(categoricals) + [1, 0]  # 1 for BonusMalus, 0 for Density and categoricals

constrained_lgbm = Pipeline(
    [
        ("lgbm", LGBMRegressor(objective="tweedie", monotone_constraints=monotone_constraints))
    ]
)

cv_constrained = GridSearchCV(
    constrained_lgbm,
    {
        "lgbm__learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        "lgbm__n_estimators": [50, 100, 150, 200],
    },
    verbose=2,
)

cv_constrained.fit(X_train_t, y_train_t, lgbm__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_train_t)

print(
    "training loss constrained_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_constrained_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss constrained_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_constrained_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# PS4 EX2
# Re-fit the best constrained LGBMRegressor
best_lgbm = cv_constrained.best_estimator_  # Best model from cross-validation
eval_results = {}  # Dictionary to store evaluation metrics

# Fit the model with evaluation sets
best_lgbm.named_steps['lgbm'].fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t,
    eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
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



# %%
# PS4 EX3
# Predictions for unconstrained model
y_pred_constrained = cv_constrained.best_estimator_.predict(X_test_t)
y_pred_unconstrained = model_pipeline.fit(X_train_t, y_train_t).predict(X_test_t)

# Evaluate metrics
metrics_constrained = evaluate_predictions(y_test_t, y_pred_constrained, sample_weight=w_test_t)
metrics_unconstrained = evaluate_predictions(y_test_t, y_pred_unconstrained, sample_weight=w_test_t)

# Print results
print("Constrained LGBM Model Metrics:\n", metrics_constrained)
print("\nUnconstrained LGBM Model Metrics:\n", metrics_unconstrained)

# PS4 EX3
# Predictions for unconstrained model
y_pred_constrained = cv_constrained.best_estimator_.predict(X_test_t)
y_pred_unconstrained = model_pipeline.fit(X_train_t, y_train_t).predict(X_test_t)

# Evaluate metrics
metrics_constrained = evaluate_predictions(y_test_t, y_pred_constrained, sample_weight=w_test_t)
metrics_unconstrained = evaluate_predictions(y_test_t, y_pred_unconstrained, sample_weight=w_test_t)

# Print results
print("Constrained LGBM Model Metrics:\n", metrics_constrained)
print("\nUnconstrained LGBM Model Metrics:\n", metrics_unconstrained)




# %%
# PS4 EX4
from dalex import Explainer

# Create DALEX Explainers
explainer_unconstrained = Explainer(
    model_pipeline.fit(X_train_t, y_train_t), 
    X_test_t, 
    y_test_t, 
    label="Unconstrained LGBM"
)

explainer_constrained = Explainer(
    cv_constrained.best_estimator_, 
    X_test_t, 
    y_test_t, 
    label="Constrained LGBM"
)

# Generate PDPs
pdp_unconstrained = explainer_unconstrained.model_profile(type="partial")
pdp_constrained = explainer_constrained.model_profile(type="partial")

# Plot PDPs for all features
pdp_unconstrained.plot(title="Partial Dependence Plot: Unconstrained LGBM")

pdp_constrained.plot(title="Partial Dependence Plot: Constrained LGBM")

# Focus on specific features (e.g., "BonusMalus" and "Density")
pdp_unconstrained_specific = explainer_unconstrained.model_profile(type="partial", variables=["BonusMalus", "Density"])
pdp_constrained_specific = explainer_constrained.model_profile(type="partial", variables=["BonusMalus", "Density"])

# Plot PDPs for specific features
pdp_unconstrained_specific.plot(title="PDP (Specific Features): Unconstrained LGBM")

pdp_constrained_specific.plot(title="PDP (Specific Features): Constrained LGBM")


# %%
