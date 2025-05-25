import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoNormal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Target
target = df["Attrition"].astype(int).values
df = df.drop(columns=["Attrition"])

# Standardize safely
X_np = df.to_numpy()
X_mean = np.nanmean(X_np, axis=0)
X_std = np.nanstd(X_np, axis=0)
X_std[X_std == 0] = 1.0
X_np = (X_np - X_mean) / X_std
X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

# Split the data
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, target, train_size=0.66, random_state=42)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

N_train, D = X_train.shape
latent_dim = 4  # Known from running PPCA seperatelt

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

def combined_ppca_blr_model(X, y=None):
    N, D = X.shape
    
    # PPCA parameters
    W = pyro.sample("W", dist.Normal(0, 1).expand([D, latent_dim]).to_event(2))
    sigma = pyro.sample("sigma", dist.HalfCauchy(1))
    
    # Logistic regression parameters
    beta = pyro.sample("beta", dist.Normal(0., 1.).expand([latent_dim]).to_event(1))
    intercept = pyro.sample("intercept", dist.Normal(0., 1.))
    
    with pyro.plate("data", N):
        z = pyro.sample("z", dist.Normal(0, 1).expand([latent_dim]).to_event(1))
        
        # Generate X
        loc = torch.matmul(z, W.mT)
        pyro.sample("obs_X", dist.Normal(loc, sigma).to_event(1), obs=X)
        
        # Generate y
        logits = (z * beta).sum(-1) + intercept
        pyro.sample("obs_y", dist.Bernoulli(logits=logits), obs=y)

def ppca_encoder(X, W, sigma):
    D, latent_dim = W.shape
    N = X.shape[0]
    WtW = W.T @ W
    cov_z_inv = WtW + sigma**2 * torch.eye(latent_dim)
    cov_z = torch.inverse(cov_z_inv)
    z_mean = (cov_z @ W.T @ X.T).T
    return z_mean

pyro.clear_param_store()
guide = AutoNormal(combined_ppca_blr_model)
optimizer = Adam({"lr": 0.01})
svi = SVI(combined_ppca_blr_model, guide, optimizer, loss=Trace_ELBO())

num_steps = 10000
for step in range(num_steps):
    loss = svi.step(X_train, y_train)
    if step % 1000 == 0:
        print(f"[Step {step}] ELBO: {loss:.2f}")

# Posterior inference using the guide (train set)
posterior_samples = guide(X_train, y_train)
W_mean = posterior_samples['W']
sigma_mean = posterior_samples['sigma']
beta_mean = posterior_samples['beta']
intercept_mean = posterior_samples['intercept']

# Use ppca_encoder for z_test
z_test_encoded = ppca_encoder(X_test, W_mean, sigma_mean)

# Predict on test set
logits_test = (z_test_encoded * beta_mean).sum(-1) + intercept_mean
probs_test = torch.sigmoid(logits_test)
y_pred_test = (probs_test > 0.5).int()

# Accuracy on test set
accuracy_test = (y_pred_test == y_test.int()).float().mean()
print(f"Test Accuracy: {accuracy_test:.3f}")

# Confusion matrix for test set
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test.numpy(), y_pred_test.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Attrition", "Attrition"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.show()