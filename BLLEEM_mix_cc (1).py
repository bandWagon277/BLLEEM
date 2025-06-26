import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp, gammaln
import arspy.ars
#import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from scipy.special import betaln

class BLLEEMGibbsSampler:
    def __init__(self, y, X, Z, P, T, L,n_iter=1000):
        """
        y: continuous health outcome (n,)
        X: three-dimensional exposure data (n, P, T)
        Z: confounder matrix (n, L+1) (including intercept term)
        P: number of exposure variables
        T: number of time points
        L: number of confounders (excluding intercept)
        """
        # dimension check
        assert X.shape == (len(y), P, T), "X dimension not match"
        assert Z.shape[1] == L+1, "Z should include intercept"
        
        self.y = y
        self.X = X
        self.Z = Z
        self.n, self.P, self.T = X.shape
        self.L = L
        self.n_iter = n_iter
        
        # calculate m
        self.m_jt = np.median(X, axis=0)  # (P, T)
        
        # Initialization
        self.beta = np.ones(L+1)  # beta_0 + beta_L
        self.alpha = np.zeros(P) 
        #self.delta = np.random.binomial(1, 0.5, P) 
        self.delta = np.ones(P)
        self.tau = np.ones(T)  # Jeffreys prior
        self.w = self._initialize_weights()  # (P, T)
        #self.w = w
        self.sigma_eps = 1.0
        self.sigma_alpha = 1.0
        self.sigma_beta = 1.0
        self.mu0 = np.zeros(L+1)
        self.q = 0.5
        self.p = np.array([0.1,0.8,0.1])
        self.eta = np.array([0.5, 0.5, 0.5])

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

    def gibbs_sample(self):
        """Perform full Gibbs sampling"""
        samples = []
        for _ in range(self.n_iter):
            self.sample_alpha_beta()
            self.sample_weights_ars()
            self.sample_delta()
            self.sample_hyperparameters()
            samples.append(self.get_state())
        return samples

    def sample_delta(self):
        
        # Update delta_j individually
        #delta_l = []
        for j in range(self.P):
            # calculate likelihood ratio
            """X_centered = self.X[:, j] - self.m_jt[j]  # (n, T)
            active_effect = self.alpha[j] * np.dot(X_centered, self.w[j])  # (n,)
            All_effect = np.dot(self.alpha,np.dot(,X_centered.T))"""
            delta_new = self.delta.copy()
            X_centered = self.X - self.m_jt
            #active_effect = self.alpha[j] * np.dot(X_centered[:,j,:], self.w[j])  # (n,)
            delta_new[j] = 0
            other_effect = np.sum(np.sum(self.w * X_centered, axis=2) * (self.alpha * delta_new), axis=1)
            #print(scale_factor)
            delta_new[j] = 1
            active_effect = np.sum(np.sum(self.w * X_centered, axis=2) * (self.alpha * delta_new), axis=1)

            
            # remove -other_effect
            log_prob1 = -0.5 * np.sum((self.y - active_effect - self.Z @ self.beta)**2) / self.sigma_eps**2
            log_prob0 = -0.5 * np.sum((self.y  - other_effect - self.Z @ self.beta)**2) / self.sigma_eps**2
            
            # add prior term
            log_prob1 += np.log(self.q)
            log_prob0 += np.log(1-self.q)
            
            # calculate normalization probability
            max_log = max(log_prob1, log_prob0)
            #prob1 = np.exp(log_prob1 - max_log) / (np.exp(log_prob1 - max_log) + np.exp(log_prob0 - max_log))
            prob1 = 1 / (1 + np.exp(-(log_prob1 - log_prob0)))
            self.delta[j] = stats.bernoulli.rvs(prob1)
            #delta_l.append(stats.bernoulli.rvs(prob1))
        #self.delta = np.array(delta_l)


    def sample_alpha_beta(self):
        len_var = int(np.sum(self.delta))
        """Joint sampling of alpha and beta"""
        # Design matrix D #variable selection
        D = np.zeros((self.n, len_var + 1 + self.L))
        for i in range(self.n):
            d_i = [self.delta[j] * np.dot(self.w[j], (self.X[i,j] - self.m_jt[j])) 
                  for j in range(self.P) if self.delta[j] == 1]
            D[i, :len_var] = d_i
            D[i, len_var:] = self.Z[i, :]

        # prior parameters
        mu_prior = np.concatenate([np.zeros(len_var), self.mu0])
        Sigma_prior_inv = np.diag(1/np.array([self.sigma_alpha**2]*len_var + [self.sigma_beta**2]*(1+self.L)))
        
        # posterior variance
        posterior_cov = np.linalg.solve(Sigma_prior_inv + (D.T @ D)/self.sigma_eps**2,np.eye(Sigma_prior_inv.shape[0]))
        
        # posterior mean
        posterior_mean = posterior_cov @ (Sigma_prior_inv @ mu_prior + (D.T @ self.y)/self.sigma_eps**2)
        
        # sampling
        theta = stats.multivariate_normal.rvs(mean=posterior_mean, cov=posterior_cov)
        self.alpha[self.delta == 1] = theta[:len_var]
        #self.alpha = np.array([1.5,-0.9,0.2,-1.1,0.8])
        self.beta = theta[len_var:]
        #self.beta = np.array([1.7,2.3,1.9])

    def _log_posterior_vjt(self, v, j, t):
        #Compute log posterior for v_{jt} (needed for ARS).
        # Compute stick-breaking weight up to time step t
        #w_jt = v_jt * np.prod(1 - self.w[j, :t]) if t > 0 else v_jt
        w_jt = v[j,t] * np.prod(1 - v[j, :t]) if t > 0 else v[j,t]
        w_new = self.w.copy()
        #w_new[j,t] = w_jt
        w_new[j] = self.stick_breaking(v[j])
        
        # Centered X values
        X_centered = self.X - self.m_jt
        # Compute residuals for likelihood term
        effect_new = np.sum(np.sum(w_new * X_centered, axis=2) * (self.alpha * self.delta), axis=1)
        residual_new = self.y - effect_new - self.Z @ self.beta
        log_likelihood = -0.5 / self.sigma_eps**2 * np.sum(residual_new**2)

        # Compute log prior term (mixture Beta distribution)
        p1, p2, p3 = self.p
        if v[j,t] == 0:
            log_prior =  np.log(p1)
        elif v[j,t] == 1:
            log_prior =  np.log(p3)
        else:
            #alpha = self.tau[t]
            alpha = np.maximum(self.tau[t],1)
            #beta = self.tau[t+1:].sum()
            beta = np.maximum(np.sum(self.tau[t:]),(self.T - t))

            log_pdf = (alpha - 1) * np.log(v[j,t]) + (beta - 1) * np.log(1 - v[j,t]) - betaln(alpha, beta)
            log_prior =  np.log(p2) + log_pdf

        return log_likelihood  + log_prior

    def _sample_vjt_cc(self, j: int, t: int):
        """Exact Gibbs step for v_{jt} using pseudo‑priors.

        Returns 0, 1, or a value in (0,1) with correct posterior weight,
        without evaluating any intractable integrals and without MH.  If the
        slab is chosen, draws the continuous value via ARS from its full
        conditional.
        """
        # Unpack mixture weights
        p1, p2, p3 = self.p  # should be updated elsewhere via Dirichlet

        # ---- Step 1: draw candidate values for each component ----
        theta0 = 0.0
        theta1 = 1.0
        theta_c = np.random.beta(self.tau[t], np.sum(self.tau[t + 1:]))  # pseudo‑prior draw

        # ---- Step 2: compute log‑likelihood for each candidate ----
        v_temp = self.v.copy()

        v_temp[j, t] = theta0
        loglik0 = self._log_posterior_vjt(v_temp, j, t)

        v_temp[j, t] = theta1
        loglik1 = self._log_posterior_vjt(v_temp, j, t)

        v_temp[j, t] = theta_c
        loglikc = self._log_posterior_vjt(v_temp, j, t)

        # ---- Step 3: assemble unnormalised log posterior for z_{jt} ----
        logpi = np.array([
            loglik0,
            loglikc,
            loglik1,
        ])
        logpi -= np.max(logpi)  # for numerical stability
        probs = np.exp(logpi)
        probs /= probs.sum()
        z = np.random.choice([0, 1, 2], p=probs)

        # ---- Step 4: return the actual v_{jt} given chosen component ----
        if z == 0:      # spike at 0
            return 0.0
        elif z == 2:    # spike at 1
            return 1.0
        else:           # slab: sample from full conditional with ARS
            def logpdf_cont(v_scalar: float):
                if v_scalar <= 0.0 or v_scalar >= 1.0:
                    return -np.inf
                v_copy = self.v.copy()
                v_copy[j, t] = v_scalar
                # likelihood part already inside _log_posterior_vjt
                return (
                    self._log_posterior_vjt(v_copy, j, t)
                    # add Beta prior log‑density
                    + (self.tau[t] - 1) * np.log(v_scalar)
                    + (np.sum(self.tau[t + 1:]) - 1) * np.log(1 - v_scalar)
                )

            v_sample = arspy.ars.adaptive_rejection_sampling(
                logpdf=logpdf_cont,
                a=1e-6,
                b=1 - 1e-6,
                domain=(1e-6, 1 - 1e-6),
                n_samples=1,
            )[0]
            return float(v_sample)

    # ------------------------------------------------------------------
    #  Main sweep over all (j,t)
    # ------------------------------------------------------------------
    def sample_weights_ars(self):
        """Gibbs sweep that plugs the new _sample_vjt_cc update."""
        """# ----- initial stick‑breaking draw (optional) -----
        self.v = np.zeros((self.P, self.T-1))
        for t in range(self.T - 1):
            self.v[:, t] = np.random.beta(self.tau[t], np.sum(self.tau[t + 1:]))

        # allocate w array
        self.w = np.zeros_like(self.v)"""
        self.v = np.zeros((self.P,self.T-1))
        for t in range(self.T-1):
            self.v[:,t] = np.random.beta(self.tau[t], np.sum(self.tau[t+1:]))

        self.v0 = self.v.copy()
        self.w0 = self.w.copy()
        # ----- iterate over features -----
        for j in range(self.P):
            if self.delta[j] == 0:
                continue  # skip pruned features
            for t in range(self.T - 1):
                self.v0[j, t] = self._sample_vjt_cc(j, t)
            # stick‑breaking to get w
            self.w0[j] = self.stick_breaking(self.v[j])
        self.v = self.v0
        self.w = self.w0

    def sample_hyperparameters(self):
        # sampling sigma_eps^2
        residual = self.y - self._predict()
        a_eps = 1 + self.n/2
        b_eps = 0.0001 + 0.5 * np.sum(residual**2)
        self.sigma_eps = 1/np.sqrt(stats.gamma.rvs(a_eps, scale=1/b_eps))

        # Update mu0's prior
        mu_0_mean = self.beta / 2
        mu_0_cov = (self.sigma_beta**2 / 2) * np.eye(self.L+1)
        self.mu0 = stats.multivariate_normal.rvs(mean=mu_0_mean, cov=mu_0_cov)

        # Posterior for sigma_beta
        beta_norm_sq = np.sum((self.beta - self.mu0)**2) + np.sum(self.mu0**2)
        shape_beta = 1 + (self.L+1) / 2
        scale_beta = 0.0001 + (1 / 2) * beta_norm_sq
        sigma_beta_inv_sq = stats.gamma.rvs(shape_beta, scale=1/scale_beta)
        self.sigma_beta = 1 / np.sqrt(sigma_beta_inv_sq)  # Convert to std deviation

        # Posterior for sigma_alpha
        alpha_norm_sq = np.sum(self.alpha**2)
        shape_alpha = 1 + self.P / 2
        scale_alpha = 0.0001 + (1 / 2) * alpha_norm_sq
        sigma_alpha_inv_sq = stats.gamma.rvs(shape_alpha, scale=1/scale_alpha)
        self.sigma_alpha = 1 / np.sqrt(sigma_alpha_inv_sq)  # Convert to std deviation"""

        """sampling delta_j ~ Bernoulli(q)"""
        # Update q's prior
        a_q = 0.5 + np.sum(self.delta)
        b_q = 0.5 + self.P - np.sum(self.delta)
        self.q = stats.beta.rvs(a_q, b_q)

        #update tau's prior
        #lambda_t = -np.log(self.w).sum(axis=0)
        #self.tau = np.random.exponential(scale=1/lambda_t)

        #update p's prior
        v = self.v.flatten()

        n0 = np.sum(v == 0)
        nB = np.sum((v > 0) & (v < 1))
        n1 = np.sum(v == 1)

        alpha_post = self.eta + np.array([n0, nB, n1])
        self.p = dirichlet.rvs(alpha_post, size=1)[0]

        #self.sigma_alpha = 1
        #self.sigma_beta = 1
        

    def _predict(self):
        """Produce predicting value"""

        X_centered = self.X - self.m_jt
        mean = np.sum(np.sum(self.w * X_centered, axis=2) * (self.alpha * self.delta), axis=1) + self.Z @ self.beta
        return mean


    def get_state(self):
        """Return current parameters"""
        return {
            'beta': self.beta.copy(),
            'alpha': self.alpha.copy(),
            'delta': self.delta.copy(),
            'w': self.w.copy(),
            'sigma_eps': self.sigma_eps,
            'sigma_alpha': self.sigma_alpha,
            'sigma_beta': self.sigma_beta,
            'mu0': self.mu0.copy(),
            'q': self.q
        }

def compute_y_mean(w,alpha,delta,X,Z,beta):
    m_jt = np.median(X, axis=0)
    X_centered = X - m_jt
    mean = np.sum(np.sum(w * X_centered, axis=2) * (alpha * delta), axis=1) + Z @ beta
    return mean

