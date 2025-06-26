import torch
import numpy as np
import scipy.stats as stats
import arspy.ars
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

class BLLEEMGibbsSampler:
    def __init__(self, y, X, Z, P, T, L, n_iter=1000, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 数据张量初始化（启用梯度）
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device, requires_grad=True)
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
        self.Z = torch.tensor(Z, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # 模型参数（显式启用梯度）
        self.beta = pyro.param("beta", torch.ones(L + 1, device=self.device, requires_grad=True))
        self.alpha = pyro.param("alpha", torch.zeros(P, device=self.device, requires_grad=True))
        self.sigma_eps = pyro.param("sigma_eps", torch.tensor(1.0, device=self.device, requires_grad=True))
        self.sigma_alpha = pyro.param("sigma_alpha", torch.tensor(1.0, device=self.device, requires_grad=True))
        self.sigma_beta = pyro.param("sigma_beta", torch.tensor(1.0, device=self.device, requires_grad=True))
        
        # 其他参数初始化
        self.delta = torch.ones(P, dtype=torch.int32, device=self.device)
        self.tau = torch.ones(T, device=self.device)
        self.m_jt = torch.median(self.X, dim=0).values.detach()  # 不追踪梯度
        self.mu0 = torch.zeros(L + 1, device=self.device)
        self.q = torch.tensor(0.5, device=self.device)
        self.n, self.P, self.T = X.shape
        self.L = L
        self.n_iter = n_iter
        
        # 初始化权重（启用梯度）
        self.w = self._initialize_weights().requires_grad_(True)
        self.v = torch.zeros((self.P, self.T), device=self.device, requires_grad=True)

    def _initialize_weights(self):
        w = torch.zeros((self.P, self.T), device=self.device)
        for j in range(self.P):
            v = torch.tensor(
                [np.random.beta(self.tau[t].item(), self.tau[t+1:].sum().item()) 
                for t in range(self.T - 1)],
                device=self.device,
                requires_grad=True  # 启用梯度
            )
            w[j] = self.stick_breaking(v)
        return w

    @staticmethod
    def stick_breaking(v):
        w = torch.zeros(len(v) + 1, device=v.device)
        remaining = 1.0
        for t in range(len(v)):
            w[t] = v[t] * remaining
            remaining *= (1 - v[t])
        w[-1] = remaining
        return w

    def gibbs_sample(self):
        samples = []
        for _ in range(self.n_iter):
            # 其他步骤禁用梯度
            with torch.no_grad():
                self.sample_alpha_beta()
                self.sample_delta()
                self.sample_hyperparameters()
            # 仅权重采样启用梯度
            self.sample_weights_ars()
            samples.append(self.get_state())
        return samples

    def sample_delta(self):
        X_centered = self.X - self.m_jt
        for j in range(self.P):
            delta_new = self.delta.clone()
            delta_new[j] = 0
            other_effect = torch.sum(torch.sum(self.w * X_centered, dim=2) * (self.alpha * delta_new), dim=1)
            delta_new[j] = 1
            active_effect = torch.sum(torch.sum(self.w * X_centered, dim=2) * (self.alpha * delta_new), dim=1)
            log_prob1 = -0.5 * torch.sum((self.y - active_effect - self.Z @ self.beta)**2) / self.sigma_eps**2
            log_prob0 = -0.5 * torch.sum((self.y - other_effect - self.Z @ self.beta)**2) / self.sigma_eps**2
            log_prob1 += torch.log(self.q)
            log_prob0 += torch.log(1 - self.q)
            prob1 = 1 / (1 + torch.exp(-(log_prob1 - log_prob0)))
            self.delta[j] = torch.bernoulli(prob1).long()

    def sample_alpha_beta(self):
        len_var = int(torch.sum(self.delta).item())
        D = torch.zeros((self.n, len_var + 1 + self.L), device=self.device)
        
        """for j in range(self.P):
            if self.delta[j]:
                X_centered_j = self.X[:, j] - self.m_jt[j]
                D[:, col_idx] = torch.matmul(X_centered_j, self.w[j])
                col_idx += 1
        D[:, len_var:] = self.Z"""
        for i in range(self.n):
            d_i = [self.delta[j] * torch.dot(self.w[j], (self.X[i, j] - self.m_jt[j]))
                for j in range(self.P) if self.delta[j] == 1]
            D[i, :len_var] = torch.tensor(d_i, dtype=D.dtype, device=D.device)
            D[i, len_var:] = self.Z[i, :]
        #------
        mu_prior = torch.cat([torch.zeros(len_var, device=self.device), self.mu0])
        Sigma_prior_inv = torch.diag(torch.cat([torch.full((len_var,), 1 / self.sigma_alpha**2, device=self.device),
                                                torch.full((1 + self.L,), 1 / self.sigma_beta**2, device=self.device)]))
        #posterior_cov = torch.linalg.inv(Sigma_prior_inv + D.T @ D / self.sigma_eps**2)
        posterior_cov = torch.linalg.solve(Sigma_prior_inv + D.T @ D / self.sigma_eps**2,
                                   torch.eye(Sigma_prior_inv.shape[0], device=D.device, dtype=D.dtype))
        posterior_mean = posterior_cov @ (Sigma_prior_inv @ mu_prior + D.T @ self.y / self.sigma_eps**2)
        dist = torch.distributions.MultivariateNormal(posterior_mean, covariance_matrix=posterior_cov)
        theta = dist.sample()
        self.alpha[self.delta == 1] = theta[:len_var]
        self.beta = theta[len_var:]

    def pyro_model(self, j, t):
        # 确保使用带梯度的参数
        beta = pyro.param("beta")
        sigma_eps = pyro.param("sigma_eps")
        alpha = pyro.param("alpha")
        
        # Beta先验参数
        tau_t = 1.0
        sum_tau_st = self.T - t
        
        # 定义采样变量（启用梯度）
        v_jt = pyro.sample(
            f"v_{j}_{t}",
            dist.Beta(tau_t, sum_tau_st).to_event(0),
            infer={"enumerate": "parallel"} if (self.delta[j] == 0).item() else {}
        )
        
        # 计算似然项（保持梯度）
        w_jt = v_jt * torch.prod(1 - self.v[j, :t]) if t > 0 else v_jt
        w_new = self.w.clone()
        w_new[j, t] = w_jt
        
        X_centered = self.X - self.m_jt
        effect_new = torch.sum(
            torch.sum(w_new * X_centered, dim=2) * (alpha * self.delta),
            dim=1
        )
        residual_new = self.y - effect_new - self.Z @ beta
        
        # 定义观测分布
        pyro.sample(
            f"obs_{j}_{t}",
            dist.Normal(0, sigma_eps).expand(residual_new.shape),
            obs=residual_new
        )

    def sample_weights_ars(self):
        # 使用中间变量初始化v
        v_init = torch.zeros((self.P, self.T), device=self.device)
        
        # Beta先验采样（无梯度）
        with torch.no_grad():
            for t in range(self.T - 1):
                alpha = self.tau[t]
                beta = self.tau[t+1:].sum()
                v_init[:, t] = dist.Beta(alpha, beta).sample((self.P,))
        
        # 将结果赋值给self.v（启用梯度）
        self.v = v_init.clone().detach().requires_grad_(True)
        self.w = torch.zeros((self.P, self.T), device=self.device, requires_grad=True)
        
        # NUTS采样循环
        for j in range(self.P):
            if self.delta[j] == 0:
                continue
            
            for t in range(self.T - 1):
                with torch.enable_grad():
                    nuts_kernel = NUTS(self.pyro_model, adapt_step_size=True)
                    mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)
                    mcmc.run(j, t)
                    
                    # 安全更新v[j, t]
                    v_new = mcmc.get_samples()[f"v_{j}_{t}"].mean()
                    self.v = torch.cat([
                        self.v[:, :t],
                        v_new.view(-1, 1),
                        self.v[:, t+1:]
                    ], dim=1).requires_grad_(True)
            
            # 更新权重
            self.w[j] = self.stick_breaking(self.v[j, :self.T-1])


    def _log_posterior_vjt(self, v, j, t):
        # 使用Pyro的约束处理边界（替代原始clamp）
        v_jt = pyro.distributions.constraints.interval(1e-6, 1-1e-6).inv(v[j, t])
        
        w_jt = v_jt * torch.prod(1 - v[j, :t]) if t > 0 else v_jt
        w_new = self.w.clone()
        w_new[j, t] = w_jt
        
        X_centered = self.X - self.m_jt
        effect_new = torch.sum(
            torch.sum(w_new * X_centered, dim=2) * (self.alpha * self.delta),
            dim=1
        )
        residual_new = self.y - effect_new - self.Z @ self.beta
        log_likelihood = -0.5 / self.sigma_eps**2 * torch.sum(residual_new**2)
        
        tau_t = 1
        sum_tau_st = (self.T - t) * tau_t
        log_prior = (tau_t - 1) * torch.log(v_jt) + (sum_tau_st - 1) * torch.log(1 - v_jt)
        
        return log_likelihood + log_prior

    

    def stick_breaking(self, v):
        w = torch.zeros(self.T, device=self.device)
        w[0] = v[0]
        for t in range(1, self.T - 1):
            w[t] = v[t] * torch.prod(1 - v[:t])
        w[-1] = 1 - torch.sum(w[:-1])
        return w
    def sample_hyperparameters(self):
        residual = self.y - self._predict()
        a_eps = 1 + self.n / 2
        b_eps = 0.0001 + 0.5 * torch.sum(residual**2)
        self.sigma_eps = 1 / torch.sqrt(torch.distributions.Gamma(a_eps, 1 / b_eps).sample().to(self.device))
        mu_0_mean = self.beta / 2
        mu_0_cov = (self.sigma_beta**2 / 2) * torch.eye(self.L + 1, device=self.device)
        dist_mu0 = torch.distributions.MultivariateNormal(mu_0_mean, covariance_matrix=mu_0_cov)
        self.mu0 = dist_mu0.sample()
        beta_norm_sq = torch.sum((self.beta - self.mu0)**2) + torch.sum(self.mu0**2)
        shape_beta = 1 + (self.L + 1) / 2
        scale_beta = 0.0001 + 0.5 * beta_norm_sq
        sigma_beta_inv_sq = torch.distributions.Gamma(shape_beta, 1 / scale_beta).sample().to(self.device)
        self.sigma_beta = 1 / torch.sqrt(sigma_beta_inv_sq)
        alpha_norm_sq = torch.sum(self.alpha**2)
        shape_alpha = 1 + self.P / 2
        scale_alpha = 0.0001 + 0.5 * alpha_norm_sq
        sigma_alpha_inv_sq = torch.distributions.Gamma(shape_alpha, 1 / scale_alpha).sample().to(self.device)
        self.sigma_alpha = 1 / torch.sqrt(sigma_alpha_inv_sq)
        a_q = 0.5 + torch.sum(self.delta).item()
        b_q = 0.5 + self.P - torch.sum(self.delta).item()
        self.q = torch.distributions.Beta(a_q, b_q).sample().to(self.device)

    def _predict(self):
        X_centered = self.X - self.m_jt
        mean = torch.sum(torch.sum(self.w * X_centered, dim=2) * (self.alpha * self.delta), dim=1)
        mean += self.Z @ self.beta
        return mean

    def get_state(self):
        return {
            'beta': self.beta.detach().cpu().numpy(),
            'alpha': self.alpha.detach().cpu().numpy(),
            'delta': self.delta.detach().cpu().numpy(),
            'w': self.w.detach().cpu().numpy(),
            'sigma_eps': self.sigma_eps.item(),
            'sigma_alpha': self.sigma_alpha.item(),
            'sigma_beta': self.sigma_beta.item(),
            'mu0': self.mu0.detach().cpu().numpy(),
            'q': self.q.item()
        }


