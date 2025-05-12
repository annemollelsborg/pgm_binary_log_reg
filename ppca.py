import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("data/train_processed.csv")
df = df.drop(columns=[
    "MonthlyIncome",
    "TotalWorkingYears",
    "PerformanceRating",
    "YearsInCurrentRole",
    "YearsWithCurrManager",
    "Department_Human Resources",
    "Department_Sales"
])

# Drop column we are predicting "Attrition"
target = df["Attrition"]
df = df.drop(columns=["Attrition"])

# Standardize safely
X_np = df.to_numpy()
X_mean = np.nanmean(X_np, axis=0)
X_std = np.nanstd(X_np, axis=0)

# Avoid division by zero
X_std[X_std == 0] = 1.0

X = (X_np - X_mean) / X_std
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = torch.tensor(X, dtype=torch.float32)

# For demonstration, let's split data into train and test sets
# Assuming target is binary and torch tensor
y = torch.tensor(target.values, dtype=torch.int64)
n_samples = X.shape[0]
train_size = int(0.8 * n_samples)
indices = torch.randperm(n_samples)
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Define Bayesian logistic regression model
def model(X, n_cat=None, y=None):
    D = X.shape[1]
    beta = pyro.sample("beta", dist.Normal(torch.zeros(D), torch.ones(D)).to_event(1))
    logits = X @ beta
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

n_steps = 5000

guides = {
    "AutoNormal": AutoNormal,
    "AutoMultivariateNormal": AutoMultivariateNormal,
    "AutoLowRankMultivariateNormal": lambda model: AutoLowRankMultivariateNormal(model, rank=2)
}

for name, GuideClass in guides.items():
    print(f"\nRunning inference with {name}")
    pyro.clear_param_store()
    guide = GuideClass(model)
    optimizer = ClippedAdam({"lr": 0.001})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)

    for step in range(n_steps):
        loss = svi.step(X_train, n_cat=None, y=y_train.float())
        if step % 1000 == 0:
            print(f"[{name} | Step {step}] ELBO: {loss:.2f}")

    predictive = Predictive(model, guide=guide, num_samples=1000, return_sites=("beta",))
    samples = predictive(X_test, n_cat=None, y=None)
    beta_samples = samples["beta"].detach().squeeze()
    beta_mean = beta_samples.mean(0)
    logits_test = X_test @ beta_mean
    probs_test = torch.sigmoid(logits_test)
    y_pred = (probs_test > 0.5).int()

    accuracy = (y_pred == y_test.int()).float().mean()
    print(f"{name} Test Accuracy: {accuracy:.3f}")

    cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Attrition", "Attrition"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix ({name})")
    plt.show()