import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .score_network import ScoreNetwork
from .utils import set_seed
from .simulator import GaussianLinearSimulator
from .utils import ve_transition_params, sample_transition, target_score, ode_integrate, kde_hpr, sample_truncated_prior, c2st_score

class TSNPSE:
    """
    Implementation of Truncated Sequential Neural Posterior Score Estimation.
    Supports a single toy simulator (GaussianLinearSimulator).  The code
    follows Algorithm 1 in the paper.
    """
    def __init__(self,
                 simulator,
                 rounds=5,
                 total_simulations=5000,
                 batch_size=256,
                 lr=1e-4,
                 device='cpu',
                 eps_hpr=0.05,
                 kde_bandwidth=0.1,
                 seed=0):
        self.sim = simulator
        self.rounds = rounds
        self.total_simulations = total_simulations
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.eps_hpr = eps_hpr
        self.kde_bandwidth = kde_bandwidth
        self.seed = seed

        self.theta_dim = simulator.dim
        self.x_dim = simulator.dim
        self.score_net = ScoreNetwork(self.theta_dim, self.x_dim).to(device)
        self.set_optim()

        # SDE parameters
        self.g, self.var_t = ve_transition_params()
        self.T = 1.0

        # Storage for datasets across rounds
        self.data_theta = []
        self.data_x = []

    def set_optim(self):
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr)

    def train_one_round(self, round_idx, n_sim):
        """Sample from prior (or truncated prior), simulate, train."""
        # prior sampler
        if round_idx == 0:
            prior_sampler = lambda n: self.sim.sample_prior(n)
        else:
            # truncated prior
            prior_sampler = lambda n: sample_truncated_prior(self.prev_prior_sampler,
                                                             self.prev_kde,
                                                             self.prev_thresh,
                                                             n)

        # Sample parameters and simulate data
        theta0 = prior_sampler(n_sim).astype(np.float32)
        x = self.sim.simulate(theta0).astype(np.float32)

        # Append to dataset
        self.data_theta.append(theta0)
        self.data_x.append(x)

        # Convert to tensors
        theta0_t = torch.from_numpy(np.concatenate(self.data_theta)).to(self.device)
        x_t = torch.from_numpy(np.concatenate(self.data_x)).to(self.device)

        # Train score network
        self.train_score_network(theta0_t, x_t)

        # After training, estimate HPR to define next truncated prior
        # Sample many posterior samples
        n_hpr = 20000
        posterior_samples = self.sample_posterior(n_hpr)
        # Estimate KDE and threshold
        thresh, kde = kde_hpr(posterior_samples, eps=self.eps_hpr,
                              bandwidth=self.kde_bandwidth)
        self.prev_thresh = thresh
        self.prev_kde = kde
        self.prev_prior_sampler = lambda n: self.sim.sample_prior(n)

    def train_score_network(self, theta0, x, epochs=10):
        self.score_net.train()
        dataset = torch.utils.data.TensorDataset(theta0, x)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_theta, batch_x in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
                batch_theta = batch_theta.to(self.device)
                batch_x = batch_x.to(self.device)
                # Sample t ~ U(0,1)
                t = torch.rand(batch_theta.size(0), 1, device=self.device)
                var = self.var_t(t)
                theta_t = sample_transition(batch_theta, t, self.var_t)
                # Target score
                t_scalar = t.squeeze()
                target = target_score(theta_t, batch_theta, t_scalar, self.var_t)
                # Network output
                pred = self.score_net(theta_t, batch_x, t)
                loss = nn.functional.mse_loss(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_theta.size(0)
            epoch_loss /= len(theta0)
            print(f'Round epoch {epoch+1}/{epochs} loss: {epoch_loss:.4f}')

    def sample_posterior(self, n_samples):
        """Sample from the current posterior by integrating the ODE."""
        self.score_net.eval()
        theta0 = torch.from_numpy(self.sim.sample_prior(n_samples).astype(np.float32)).to(self.device)
        # Use a simple ODE integrator (Euler) with 100 steps
        with torch.no_grad():
            theta = theta0.clone()
            steps = 100
            dt = -self.T / steps
            for _ in range(steps):
                t = torch.full((theta.size(0), 1), self.T, device=self.device)
                s = self.score_net(theta, torch.zeros_like(theta), t)
                v = -0.5 * self.var_t(self.T) * s  # VE SDE
                theta = theta + v * dt
                self.T += dt
            return theta.cpu().numpy()

    def evaluate(self, n_true_samples=10000, n_est_samples=10000):
        """Compute C2ST score between true and estimated posterior."""
        # True posterior samples (analytic for Gaussian linear)
        true_means, true_cov = self.sim.analytical_posterior(self.sim.sample_prior(1)[0])
        true_samples = np.random.multivariate_normal(true_means, true_cov,
                                                     size=n_true_samples)
        # Estimated posterior samples
        est_samples = self.sample_posterior(n_est_samples)
        acc = c2st_score(true_samples, est_samples)
        print(f'C2ST accuracy (higher is worse): {acc:.4f}')
        return acc

    def run(self):
        sims_per_round = self.total_simulations // self.rounds
        for r in range(self.rounds):
            print(f'=== Round {r+1}/{self.rounds} ===')
            self.train_one_round(r, sims_per_round)
        print("Training finished. Evaluation:")
        self.evaluate()