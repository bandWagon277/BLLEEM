import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp, gammaln
import arspy.ars
#import matplotlib.pyplot as plt

class BLLEEMBinaryGibbsSampler:
    def __init__(self, y_binary, X, Z, P, T, L, n_iter=1000):
        """
        y_binary: binary health outcome (n,) - values should be 0 or 1
        X: three-dimensional exposure data (n, P, T)
        Z: confounder matrix (n, L+1) (including intercept term)
        P: number of exposure variables
        T: number of time points
        L: number of confounders (excluding intercept)
        """
        # dimension check
        assert X.shape == (len(y_binary), P, T), "X dimension not match"
        assert Z.shape[1] == L+1, "Z should include intercept"
        assert np.all(np.isin(y_binary, [0, 1])), "y_binary should contain only 0 and 1"
        
        self.y_binary = y_binary
        self.X = X
        self.Z = Z
        self.n, self.P, self.T = X.shape
        self.L = L
        self.n_iter = n_iter
        
        # Initialize latent variable y_star
        # Initialize with reasonable values based on observed binary outcomes
        """self.y_star = np.where(y_binary == 1, 
                              np.random.normal(0.5, 0.5, self.n), 
                              np.random.normal(-0.5, 0.5, self.n))"""

        def truncated_normal(mean, std, lower, upper, size):
            a, b = (lower - mean) / std, (upper - mean) / std
            return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

        self.y_star = np.empty(self.n)

        # y_binary == 1
        idx1 = y_binary == 1
        self.y_star[idx1] = truncated_normal(mean=0.5, std=1, lower=0, upper=np.inf, size=np.sum(idx1))
        # y_binary == 0
        idx0 = y_binary == 0
        self.y_star[idx0] = truncated_normal(mean=-0.5, std=1, lower=-np.inf, upper=0, size=np.sum(idx0))

        
        # calculate m
        self.m_jt = np.median(X, axis=0)  # (P, T)
        
        # Initialization - same as original
        self.beta = np.ones(L+1)  # beta_0 + beta_L
        self.alpha = np.zeros(P) 
        self.delta = np.ones(P)
        self.tau = np.ones(T)  # Jeffreys prior
        self.w = self._initialize_weights()  # (P, T)
        self.sigma_eps = 1.0
        self.sigma_alpha = 1.0
        self.sigma_beta = 1.0
        self.mu0 = np.zeros(L+1)
        self.q = 0.5

    def _initialize_weights(self):
        """Initialize stick-breaking weight w"""
        w = np.zeros((self.P, self.T))
        v = np.zeros(self.T-1)
        for j in range(self.P):
            for t in range(self.T-1):
                v[t] = np.random.beta(self.tau[t], np.sum(self.tau[t+1:]))
            w[j] = self.stick_breaking(v)
        return w

    @staticmethod
    def stick_breaking(v):
        """Convert Beta variables to stick-breaking weights"""
        w = np.zeros(len(v)+1)
        remaining = 1.0
        for t in range(len(v)):
            w[t] = v[t] * remaining
            remaining *= (1 - v[t])
        w[-1] = remaining
        return w

    def sample_y_star(self):
        """
        Sample latent variable y_star from truncated normal distribution
        y_star | Others, y_binary ~ TN(mu_i, sigma_eps^2, bounds)
        where bounds = (0, inf) if y_binary = 1, (-inf, 0] if y_binary = 0
        """
        # Compute mean for each observation
        mu = self._predict()
        
        for i in range(self.n):
            if self.y_binary[i] == 1:
                # y_star > 0, sample from truncated normal (0, inf)
                self.y_star[i] = stats.truncnorm.rvs(
                    a=(0 - mu[i]) / self.sigma_eps,  # standardized lower bound
                    b=np.inf,                        # standardized upper bound
                    loc=mu[i],                       # mean
                    scale=self.sigma_eps             # std
                )
            else:
                # y_star <= 0, sample from truncated normal (-inf, 0]
                self.y_star[i] = stats.truncnorm.rvs(
                    a=-np.inf,                       # standardized lower bound
                    b=(0 - mu[i]) / self.sigma_eps,  # standardized upper bound
                    loc=mu[i],                       # mean
                    scale=self.sigma_eps             # std
                )

    def gibbs_sample(self):
        """Perform full Gibbs sampling for binary outcomes"""
        samples = []
        for iteration in range(self.n_iter):
            # Sample latent variable first
            self.sample_y_star()
            
            # Then sample other parameters using y_star (similar to continuous case)
            self.sample_alpha_beta()
            self.sample_weights_ars()
            self.sample_delta()
            self.sample_hyperparameters()
            
            # Store samples
            samples.append(self.get_state())
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}")
                
        return samples

    def sample_delta(self):
        """Sample delta using y_star instead of y"""
        for j in range(self.P):
            delta_new = self.delta.copy()
            X_centered = self.X - self.m_jt
            
            # Calculate effects with delta_j = 0
            delta_new[j] = 0
            other_effect = np.sum(np.sum(self.w * X_centered, axis=2) * (self.alpha * delta_new), axis=1)
            
            # Calculate effects with delta_j = 1
            delta_new[j] = 1
            active_effect = np.sum(np.sum(self.w * X_centered, axis=2) * (self.alpha * delta_new), axis=1)
            
            # Use y_star instead of y for likelihood calculation
            log_prob1 = -0.5 * np.sum((self.y_star - active_effect - self.Z @ self.beta)**2) / self.sigma_eps**2
            log_prob0 = -0.5 * np.sum((self.y_star - other_effect - self.Z @ self.beta)**2) / self.sigma_eps**2
            
            # Add prior term
            log_prob1 += np.log(self.q)
            log_prob0 += np.log(1-self.q)
            
            # Calculate probability
            prob1 = 1 / (1 + np.exp(-(log_prob1 - log_prob0)))
            self.delta[j] = stats.bernoulli.rvs(prob1)

    def sample_alpha_beta(self):
        """Joint sampling of alpha and beta using y_star"""
        len_var = int(np.sum(self.delta))
        
        # Design matrix D - variable selection
        D = np.zeros((self.n, len_var + 1 + self.L))
        for i in range(self.n):
            d_i = [self.delta[j] * np.dot(self.w[j], (self.X[i,j] - self.m_jt[j])) 
                  for j in range(self.P) if self.delta[j] == 1]
            D[i, :len_var] = d_i
            D[i, len_var:] = self.Z[i, :]

        # Prior parameters
        mu_prior = np.concatenate([np.zeros(len_var), self.mu0])
        Sigma_prior_inv = np.diag(1/np.array([self.sigma_alpha**2]*len_var + [self.sigma_beta**2]*(1+self.L)))
        
        # Posterior variance
        posterior_cov = np.linalg.solve(Sigma_prior_inv + (D.T @ D)/self.sigma_eps**2, np.eye(Sigma_prior_inv.shape[0]))
        
        # Posterior mean - use y_star instead of y
        posterior_mean = posterior_cov @ (Sigma_prior_inv @ mu_prior + (D.T @ self.y_star)/self.sigma_eps**2)
        
        # Sampling
        theta = stats.multivariate_normal.rvs(mean=posterior_mean, cov=posterior_cov)
        self.alpha[self.delta == 1] = theta[:len_var]
        self.beta = theta[len_var:]

    def sample_weights_ars(self):
        """Perform ARS-based stick-breaking sampling - same as original but uses y_star"""
        self.v = np.zeros((self.P, self.T))
        for t in range(self.T-1):
            self.v[:,t] = np.random.beta(self.tau[t], np.sum(self.tau[t+1:]))

        self.v0 = self.v.copy()
        self.w0 = self.w.copy()
        
        for j in range(self.P):
            if self.delta[j] == 0:
                continue  # Skip non-selected features

            # Sample v_{jt} using ARS
            v = np.zeros(self.T - 1)
            for t in range(self.T - 1):
                v[t] = self._sample_vjt_ars(j, t)
                self.v0[j,t] = v[t]

            # Compute w_jt using stick-breaking process
            self.w0[j] = self.stick_breaking(v)
            
        self.v = self.v0
        self.w = self.w0

    def _log_posterior_vjt(self, v, j, t):
        """Compute log posterior for v_{jt} - modified to use y_star"""
        w_jt = v[j,t] * np.prod(1 - v[j, :t]) if t > 0 else v[j,t]
        w_new = self.w.copy()
        w_new[j,t] = w_jt
        
        # Centered X values
        X_centered = self.X - self.m_jt
        
        # Compute residuals for likelihood term - use y_star
        effect_new = np.sum(np.sum(w_new * X_centered, axis=2) * (self.alpha * self.delta), axis=1)
        residual_new = self.y_star - effect_new - self.Z @ self.beta
        log_likelihood = -0.5 / self.sigma_eps**2 * np.sum(residual_new**2)

        # Compute log prior term (Beta distribution)
        tau_t = 1
        sum_tau_st = (self.T - t)
        log_prior = (
            (tau_t - 1) * np.log(v[j,t] + 1e-10)
            + (sum_tau_st - 1) * np.log(1 - v[j,t] + 1e-10)
        )

        return log_likelihood + log_prior

    def _sample_vjt_ars(self, j, t):
        """Use Adaptive Rejection Sampling (ARS) to sample v_{jt}"""
        lb, ub = 1e-6, 1 - 1e-6
        
        def logpdf(v_scalar):
            v_copy = self.v.copy()
            v_copy[j, t] = v_scalar
            return self._log_posterior_vjt(v_copy, j, t)
        
        v_jt_sampled = arspy.ars.adaptive_rejection_sampling(
            logpdf=logpdf,
            a=lb,
            b=ub,
            domain=(lb, ub),
            n_samples=1
        )

        return v_jt_sampled[0]

    def sample_hyperparameters(self):
        """Sample hyperparameters using y_star"""
        # Sample sigma_eps^2 - use y_star
        residual = self.y_star - self._predict()
        a_eps = 1 + self.n/2
        b_eps = 0.0001 + 0.5 * np.sum(residual**2)
        #self.sigma_eps = 1/np.sqrt(stats.gamma.rvs(a_eps, scale=1/b_eps))

        # Update mu0's prior
        mu_0_mean = self.beta / 2
        mu_0_cov = (self.sigma_beta**2 / 2) * np.eye(self.L+1)
        self.mu0 = stats.multivariate_normal.rvs(mean=mu_0_mean, cov=mu_0_cov)

        # Posterior for sigma_beta
        beta_norm_sq = np.sum((self.beta - self.mu0)**2) + np.sum(self.mu0**2)
        shape_beta = 1 + (self.L+1) / 2
        scale_beta = 0.0001 + (1 / 2) * beta_norm_sq
        sigma_beta_inv_sq = stats.gamma.rvs(shape_beta, scale=1/scale_beta)
        self.sigma_beta = 1 / np.sqrt(sigma_beta_inv_sq)

        # Posterior for sigma_alpha
        alpha_norm_sq = np.sum(self.alpha**2)
        shape_alpha = 1 + self.P / 2
        scale_alpha = 0.0001 + (1 / 2) * alpha_norm_sq
        sigma_alpha_inv_sq = stats.gamma.rvs(shape_alpha, scale=1/scale_alpha)
        self.sigma_alpha = 1 / np.sqrt(sigma_alpha_inv_sq)

        # Update q (prior for delta)
        a_q = 0.5 + np.sum(self.delta)
        b_q = 0.5 + self.P - np.sum(self.delta)
        self.q = stats.beta.rvs(a_q, b_q)

    def _predict(self):
        """Produce predicting value for y_star"""
        X_centered = self.X - self.m_jt
        mean = np.sum(np.sum(self.w * X_centered, axis=2) * (self.alpha * self.delta), axis=1) + self.Z @ self.beta
        return mean

    def predict_probability(self):
        """Predict probability P(y=1) using current parameters"""
        mu = self._predict()
        return stats.norm.cdf(mu / self.sigma_eps)

    def get_state(self):
        """Return current parameters including y_star"""
        return {
            'beta': self.beta.copy(),
            'alpha': self.alpha.copy(),
            'delta': self.delta.copy(),
            'w': self.w.copy(),
            'y_star': self.y_star.copy(),  # Include latent variable
            'sigma_eps': self.sigma_eps,
            'sigma_alpha': self.sigma_alpha,
            'sigma_beta': self.sigma_beta,
            'mu0': self.mu0.copy(),
            'q': self.q
        }

def compute_y_mean_binary(w, alpha, delta, X, Z, beta):
    """Compute mean of latent variable y_star"""
    m_jt = np.median(X, axis=0)
    X_centered = X - m_jt
    mean = np.sum(np.sum(w * X_centered, axis=2) * (alpha * delta), axis=1) + Z @ beta
    return mean

def compute_y_prob(w, alpha, delta, X, Z, beta, sigma_eps=1.0):
    """Compute probability P(y=1) = P(y_star > 0)"""
    mu = compute_y_mean_binary(w, alpha, delta, X, Z, beta)
    return stats.norm.cdf(mu / sigma_eps)

def plot_trace_binary(samples, var_name):
    """Plot trace for binary sampler"""
    plt.figure(figsize=(10, 4))
    plt.plot(samples, alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel(var_name)
    plt.title(f"Trace Plot for {var_name}")
    plt.show()

"""# Example usage
if __name__ == "__main__":
    # Generate simulated binary data
    np.random.seed(123)
    n, P, T, L = 100, 3, 3, 2
    
    # Generate exposure and confounder data
    X = np.random.normal(loc=0, scale=1, size=(n, P, T))
    Z = np.hstack([np.ones((n,1)), np.random.normal(size=(n, L))])
    
    # True parameters for simulation
    w_true = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
    alpha_true = np.array([1.0, -0.8, 0.6])
    delta_true = np.array([1, 1, 0])  # Third exposure not selected
    beta_true = np.array([0.5, 0.3, -0.4])
    
    # Generate latent variable y_star
    y_star_true = compute_y_mean_binary(w_true, alpha_true, delta_true, X, Z, beta_true)
    y_star_true += np.random.normal(0, 1, n)  # Add noise
    
    # Generate binary outcome
    y_binary = (y_star_true > 0).astype(int)
    
    print(f"Binary outcome distribution: {np.bincount(y_binary)}")
    
    # Fit the model
    sampler = BLLEEMBinaryGibbsSampler(y_binary, X, Z, P, T, L, n_iter=1000)
    samples = sampler.gibbs_sample()
    
    print("\nFinal state:")
    print(samples[-1])"""