## binary logistic regression, using PPCA data set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
import seaborn as sns
import torch
import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam
import itertools
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer import Predictive
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

palette = itertools.cycle(sns.color_palette())

reduced_data = pd.read_csv('data/reduced_dataset.csv')

X = reduced_data.iloc[:, :4].values
y = reduced_data.iloc[:, 4].astype(int).values
ind = y.copy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("ind shape:", ind.shape)

train_perc = 0.66 # percentage of training data
split_point = int(train_perc*len(y))
perm = np.random.permutation(len(y)) # we also randomize the dataset
ix_train = perm[:split_point]
ix_test = perm[split_point:]
X_train = X[ix_train,:]
X_test = X[ix_test,:]
y_train = y[ix_train]
y_test = y[ix_test]
print("num train: %d" % len(y_train))
print("num test: %d" % len(y_test))

# Model definition
def model(X, n_cat, y=None):
    n_features = X.shape[1]

    beta = pyro.sample("beta", dist.Laplace(0., 1.).expand([n_features, 1]).to_event(2)) # Priors, accuracy = .575
    intercept = pyro.sample("intercept", dist.Normal(0., 1.)) # Intercept, accuracy = .578
    logits = X @ beta + intercept  # shape: (N, 1), input features to the bernouli

    with pyro.plate("data", X.shape[0]):
        y = pyro.sample("y", dist.Bernoulli(logits=logits.squeeze(-1)), obs=y)
    
    return y

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Clear previous state
pyro.clear_param_store()

from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal

guides = {
    "AutoNormal": AutoNormal,
    "AutoMultivariateNormal": AutoMultivariateNormal,
    "AutoLowRankMultivariateNormal": lambda model: AutoLowRankMultivariateNormal(model, rank=2)
}

n_steps = 40000

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