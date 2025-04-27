import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoNormal

# Load dataset
df = pd.read_csv("/Data/train_processed.csv")

# Keep only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Remove constant columns
df_numeric = df_numeric.loc[:, df_numeric.std() > 0]

# Standardize
X = (df_numeric - df_numeric.mean()) / df_numeric.std()
X = X.fillna(0)  # Just to be safe
X = torch.tensor(X.values, dtype=torch.float32)

# Set latent dimension (choose 2 for now, or you can change later)
latent_dim = 2

# Define PPCA model
def ppca_model(X, latent_dim):
    N, D = X.shape
    W = pyro.sample("W", dist.Normal(0, 1).expand([D, latent_dim]).to_event(2))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.0))
    
    with pyro.plate("data", N):
        z = pyro.sample("z", dist.Normal(0, 1).expand([latent_dim]).to_event(1))
        loc = torch.matmul(z, W.T)
        pyro.sample("obs", dist.Normal(loc, sigma).to_event(1), obs=X)

import matplotlib.pyplot as plt

# Automatic search for best latent dimension
best_latent_dim = None
best_elbo = float('-inf')
best_z_loc = None

elbos = []
dims = list(range(1, 21))  # Try dimensions from 1 to 20

for latent_dim in dims:
    pyro.clear_param_store()
    guide = AutoNormal(ppca_model)
    optimizer = Adam({"lr": 0.01})
    svi = SVI(ppca_model, guide, optimizer, loss=Trace_ELBO())

    num_steps = 1000
    final_loss = None
    for step in range(num_steps):
        loss = svi.step(X, latent_dim)
        final_loss = loss
    
    elbo = -final_loss  # ELBO is negative of loss
    elbos.append(elbo)

    if elbo > best_elbo:
        best_elbo = elbo
        best_latent_dim = latent_dim
        best_z_loc = pyro.param("AutoNormal.locs.z").detach().numpy()

# Plot ELBO vs latent dimensions
plt.plot(dims, elbos, marker='o')
plt.xlabel('Latent Dimensions')
plt.ylabel('ELBO')
plt.title('ELBO vs Latent Dimensionality')
plt.grid(True)
plt.show()

print(f"Best latent dimension selected: {best_latent_dim}")

# Save reduced dataset
reduced_df = pd.DataFrame(best_z_loc, columns=[f"z{i+1}" for i in range(best_latent_dim)])
reduced_df.to_csv("/Data/reduced_dataset.csv", index=False)

print("Reduced dataset saved")