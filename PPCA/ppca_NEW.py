import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoNormal
import matplotlib.pyplot as plt

np.random.seed(42)

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

# Set latent dimension
latent_dim = 2

# Define PPCA model
def ppca_model(X, latent_dim):
    N, D = X.shape
    W = pyro.sample("W", dist.Normal(0, 1).expand([D, latent_dim]).to_event(2))
    sigma  = pyro.sample("sigma",  dist.HalfCauchy(1)) 
    
    with pyro.plate("data", N):
        z = pyro.sample("z", dist.Normal(0, 1).expand([latent_dim]).to_event(1))
        loc = torch.matmul(z, W.T)
        pyro.sample("obs", dist.Normal(loc, sigma).to_event(1), obs=X)

# Search for best latent dimension
best_latent_dim = None
best_elbo = float('-inf')
best_z_loc = None

elbos = []
dims = list(range(1, 11))  # Dimensions from 1 to 10

for latent_dim in dims:
    pyro.clear_param_store()
    
    def model_wrapped(X):
        return ppca_model(X, latent_dim)

    guide = AutoNormal(model_wrapped)
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model_wrapped, guide, optimizer, loss=Trace_ELBO())
    
    num_steps = 1000
    final_loss = None
    for step in range(num_steps):
        loss = svi.step(X)
        final_loss = loss
    elbo = -final_loss
    elbos.append(elbo)

    if elbo > best_elbo:
        best_elbo = elbo
        best_latent_dim = latent_dim
        best_z_loc = guide(X)["z"].detach().numpy()

# Plot ELBO vs latent dimensions
plt.plot(dims, elbos, marker='o')
plt.xlabel('Latent Dimensions')
plt.ylabel('ELBO')
plt.title('ELBO vs Latent Dimensionality')
plt.grid(True)
plt.show()

print(f"Best latent dimension selected: {best_latent_dim}")

# Save reduced dataset and merge with target
reduced_df = pd.DataFrame(best_z_loc, columns=[f"z{i+1}" for i in range(best_latent_dim)])
reduced_df["Attrition"] = target.values
reduced_df.to_csv("data/reduced_dataset.csv", index=False)

print("Reduced dataset saved")