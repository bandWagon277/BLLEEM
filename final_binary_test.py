import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp, gammaln
import arspy.ars
#import matplotlib.pyplot as plt

class ImprovedBLLEEMBinaryGibbsSampler:
    def __init__(self, y_binary, X, Z, P, T, L, n_iter=1000):
        """
        改进的BLLEEM Binary Gibbs采样器
        """
        # 维度检查
        assert X.shape == (len(y_binary), P, T), "X dimension not match"
        assert Z.shape[1] == L+1, "Z should include intercept"
        assert np.all(np.isin(y_binary, [0, 1])), "y_binary should contain only 0 and 1"
        
        self.y_binary = y_binary
        self.X = X
        self.Z = Z
        self.n, self.P, self.T = X.shape
        self.L = L
        self.n_iter = n_iter
        
        # 改进的初始化策略
        self._improved_initialization()
        
        # 计算中位数
        self.m_jt = np.median(X, axis=0)  # (P, T)
        
        # 添加自适应步长
        self.acceptance_rates = {'alpha_beta': [], 'weights': [], 'delta': []}
        self.adapt_interval = 50

    def _improved_initialization(self):
        """改进的参数初始化"""
        # 更好的y_star初始化：基于简单logistic回归
        try:
            from sklearn.linear_model import LogisticRegression
            # 使用X的简化版本进行初始拟合
            X_flat = self.X.reshape(self.n, -1)
            lr = LogisticRegression(fit_intercept=False, max_iter=100)
            lr.fit(np.hstack([X_flat, self.Z[:, 1:]]), self.y_binary)  # 排除intercept
            initial_prob = lr.predict_proba(np.hstack([X_flat, self.Z[:, 1:]]))[:, 1]
            
            # 基于预测概率初始化y_star
            self.y_star = np.where(self.y_binary == 1,
                                  np.random.normal(np.maximum(initial_prob, 0.1), 0.5, self.n),
                                  np.random.normal(np.minimum(initial_prob - 1, -0.1), 0.5, self.n))
        except ImportError:
            # 备选方案：更保守的初始化
            self.y_star = np.where(self.y_binary == 1,
                                  np.random.normal(1.0, 0.3, self.n),
                                  np.random.normal(-1.0, 0.3, self.n))
        
        # 参数初始化
        self.beta = np.random.normal(0, 0.5, self.L + 1)
        self.alpha = np.random.normal(0, 0.5, self.P)
        
        # 更保守的delta初始化
        self.delta = np.random.binomial(1, 0.3, self.P)  # 降低初始选择概率
        
        self.tau = np.ones(self.T)
        self.w = self._initialize_weights_improved()
        self.sigma_eps = 1.0
        self.sigma_alpha = 1.0
        self.sigma_beta = 1.0
        self.mu0 = np.zeros(self.L + 1)
        self.q = 0.3  # 降低先验选择概率

    def _initialize_weights_improved(self):
        """改进的权重初始化"""
        w = np.zeros((self.P, self.T))
        for j in range(self.P):
            # 更平滑的初始权重
            raw_weights = np.random.dirichlet(np.ones(self.T) * 2, 1)[0]
            w[j] = raw_weights
        return w

    def sample_y_star_improved(self):
        """改进的latent variable采样"""
        mu = self._predict()
        
        # 使用向量化的truncated normal采样
        mask_pos = (self.y_binary == 1)
        mask_neg = (self.y_binary == 0)
        
        if np.any(mask_pos):
            # y_star > 0的情况
            a_pos = (0 - mu[mask_pos]) / self.sigma_eps
            self.y_star[mask_pos] = stats.truncnorm.rvs(
                a=a_pos, b=np.inf, 
                loc=mu[mask_pos], scale=self.sigma_eps,
                size=np.sum(mask_pos)
            )
        
        if np.any(mask_neg):
            # y_star <= 0的情况
            b_neg = (0 - mu[mask_neg]) / self.sigma_eps
            self.y_star[mask_neg] = stats.truncnorm.rvs(
                a=-np.inf, b=b_neg,
                loc=mu[mask_neg], scale=self.sigma_eps,
                size=np.sum(mask_neg)
            )

    def sample_delta_improved(self):
        """改进的变量选择采样"""
        X_centered = self.X - self.m_jt
        
        for j in range(self.P):
            # 计算其他变量的贡献
            other_indices = np.arange(self.P) != j
            other_effect = np.sum(np.sum(self.w[other_indices] * X_centered[:, other_indices], axis=2) * 
                                (self.alpha[other_indices] * self.delta[other_indices]), axis=1)
            
            # j变量的贡献
            j_contribution = np.sum(self.w[j] * X_centered[:, j], axis=1)
            
            # 计算条件似然
            residual_base = self.y_star - other_effect - self.Z @ self.beta
            
            # delta_j = 0的情况
            log_lik0 = -0.5 * np.sum(residual_base**2) / self.sigma_eps**2
            
            # delta_j = 1的情况
            residual_active = residual_base - self.alpha[j] * j_contribution
            log_lik1 = -0.5 * np.sum(residual_active**2) / self.sigma_eps**2
            
            # 加入先验
            log_prior1 = np.log(self.q + 1e-10)
            log_prior0 = np.log(1 - self.q + 1e-10)
            
            # 计算后验概率
            log_odds = (log_lik1 + log_prior1) - (log_lik0 + log_prior0)
            prob1 = 1.0 / (1.0 + np.exp(-log_odds))
            
            # 数值稳定性检查
            prob1 = np.clip(prob1, 1e-10, 1-1e-10)
            
            self.delta[j] = stats.bernoulli.rvs(prob1)

    def sample_alpha_beta_improved(self):
        """改进的联合采样"""
        active_vars = np.where(self.delta == 1)[0]
        n_active = len(active_vars)
        
        if n_active == 0:
            # 如果没有选中的变量，只采样beta
            prior_cov = self.sigma_beta**2 * np.eye(self.L + 1)
            post_cov = np.linalg.inv(np.linalg.inv(prior_cov) + 
                                   (self.Z.T @ self.Z) / self.sigma_eps**2)
            post_mean = post_cov @ (np.linalg.inv(prior_cov) @ self.mu0 + 
                                  (self.Z.T @ self.y_star) / self.sigma_eps**2)
            self.beta = stats.multivariate_normal.rvs(post_mean, post_cov)
            return
        
        # 构建设计矩阵
        X_centered = self.X - self.m_jt
        D = np.zeros((self.n, n_active + self.L + 1))
        
        for i, j in enumerate(active_vars):
            D[:, i] = np.sum(self.w[j] * X_centered[:, j], axis=1)
        
        D[:, n_active:] = self.Z
        
        # 先验协方差
        prior_cov_diag = np.concatenate([
            np.full(n_active, self.sigma_alpha**2),
            np.full(self.L + 1, self.sigma_beta**2)
        ])
        prior_cov_inv = np.diag(1.0 / prior_cov_diag)
        
        # 先验均值
        prior_mean = np.concatenate([np.zeros(n_active), self.mu0])
        
        # 后验参数
        try:
            post_precision = prior_cov_inv + (D.T @ D) / self.sigma_eps**2
            post_cov = np.linalg.inv(post_precision)
            post_mean = post_cov @ (prior_cov_inv @ prior_mean + 
                                  (D.T @ self.y_star) / self.sigma_eps**2)
            
            # 采样
            theta = stats.multivariate_normal.rvs(post_mean, post_cov)
            
            # 更新参数
            self.alpha[active_vars] = theta[:n_active]
            self.alpha[self.delta == 0] = 0  # 确保非选中变量的系数为0
            self.beta = theta[n_active:]
            
        except np.linalg.LinAlgError:
            # 数值稳定性问题的备选方案
            print("Warning: Numerical instability in alpha_beta sampling")
            self.alpha[active_vars] += np.random.normal(0, 0.01, n_active)
            self.beta += np.random.normal(0, 0.01, self.L + 1)

    def sample_hyperparameters_improved(self):
        """改进的超参数采样"""
        # 采样sigma_eps
        residual = self.y_star - self._predict()
        a_eps = 2 + self.n / 2  # 更保守的先验
        b_eps = 1 + 0.5 * np.sum(residual**2)
        
        # 确保数值稳定性
        sigma_eps_sq = 1.0 / stats.gamma.rvs(a_eps, scale=1/b_eps)
        self.sigma_eps = np.sqrt(np.maximum(sigma_eps_sq, 1e-6))
        
        # 采样其他超参数（类似的改进）
        # mu0
        mu_cov = (self.sigma_beta**2 / 2) * np.eye(self.L + 1)
        mu_mean = self.beta / 2
        self.mu0 = stats.multivariate_normal.rvs(mu_mean, mu_cov)
        
        # sigma_beta
        beta_ss = np.sum((self.beta - self.mu0)**2) + np.sum(self.mu0**2)
        shape_beta = 2 + (self.L + 1) / 2
        scale_beta = 1 + beta_ss / 2
        sigma_beta_sq = 1.0 / stats.gamma.rvs(shape_beta, scale=1/scale_beta)
        self.sigma_beta = np.sqrt(np.maximum(sigma_beta_sq, 1e-6))
        
        # sigma_alpha
        alpha_ss = np.sum(self.alpha**2)
        shape_alpha = 2 + self.P / 2
        scale_alpha = 1 + alpha_ss / 2
        sigma_alpha_sq = 1.0 / stats.gamma.rvs(shape_alpha, scale=1/scale_alpha)
        self.sigma_alpha = np.sqrt(np.maximum(sigma_alpha_sq, 1e-6))
        
        # q
        a_q = 1 + np.sum(self.delta)
        b_q = 1 + self.P - np.sum(self.delta)
        self.q = stats.beta.rvs(a_q, b_q)

    def gibbs_sample_improved(self):
        """改进的Gibbs采样主循环"""
        samples = []
        
        for iteration in range(self.n_iter):
            # 采样顺序很重要
            self.sample_y_star_improved()
            self.sample_delta_improved()
            self.sample_alpha_beta_improved()
            
            # 简化权重采样（暂时使用固定权重或简单更新）
            if iteration % 10 == 0:  # 降低权重更新频率
                self._simple_weight_update()
            
            self.sample_hyperparameters_improved()
            
            # 存储样本
            samples.append(self.get_state())
            
            if iteration % 100 == 0:
                active_vars = np.sum(self.delta)
                print(f"Iteration {iteration}, Active vars: {active_vars}, "
                      f"sigma_eps: {self.sigma_eps:.4f}")
                
        return samples

    def _simple_weight_update(self):
        """简化的权重更新"""
        # 对于每个选中的变量，轻微调整权重
        for j in range(self.P):
            if self.delta[j] == 1:
                # 添加小的随机扰动
                noise = np.random.dirichlet(np.ones(self.T) * 10)  # 高concentration保持接近原值
                self.w[j] = 0.9 * self.w[j] + 0.1 * noise
            else:
                # 对于未选中的变量，权重趋向均匀
                self.w[j] = 0.8 * self.w[j] + 0.2 * np.ones(self.T) / self.T

    def _predict(self):
        """预测函数"""
        X_centered = self.X - self.m_jt
        mean = np.sum(np.sum(self.w * X_centered, axis=2) * 
                     (self.alpha * self.delta), axis=1) + self.Z @ self.beta
        return mean

    def get_state(self):
        """获取当前状态"""
        return {
            'beta': self.beta.copy(),
            'alpha': self.alpha.copy(),
            'delta': self.delta.copy(),
            'w': self.w.copy(),
            'y_star': self.y_star.copy(),
            'sigma_eps': self.sigma_eps,
            'sigma_alpha': self.sigma_alpha,
            'sigma_beta': self.sigma_beta,
            'mu0': self.mu0.copy(),
            'q': self.q
        }

# 使用示例
if __name__ == "__main__":
    # 测试改进版本
    np.random.seed(123)
    n, P, T, L = 100, 10, 3, 2
    
    # 生成测试数据
    X = np.random.normal(0, 1, (n, P, T))
    Z = np.hstack([np.ones((n, 1)), np.random.normal(0, 1, (n, L))])
    
    # 简单的真实参数
    w_true = np.random.dirichlet(np.ones(T), P)
    alpha_true = np.random.normal(0, 1, P)
    delta_true = np.random.binomial(1, 0.3, P)
    alpha_true *= delta_true  # 只有选中的变量有非零系数
    beta_true = np.random.normal(0, 1, L + 1)
    
    # 生成二元结果
    m_jt = np.median(X, axis=0)
    X_centered = X - m_jt
    y_star_true = (np.sum(np.sum(w_true * X_centered, axis=2) * 
                         (alpha_true * delta_true), axis=1) + 
                  Z @ beta_true + np.random.normal(0, 1, n))
    y_binary = (y_star_true > 0).astype(int)
    
    print(f"True delta: {delta_true}")
    print(f"Binary outcome distribution: {np.bincount(y_binary)}")
    
    # 拟合改进的模型
    sampler = ImprovedBLLEEMBinaryGibbsSampler(y_binary, X, Z, P, T, L, n_iter=500)
    samples = sampler.gibbs_sample_improved()
    
    # 分析结果
    burn_in = len(samples) // 2
    delta_posterior = np.mean([s['delta'] for s in samples[burn_in:]], axis=0)
    print(f"Posterior delta probability: {delta_posterior}")
    print(f"Estimated delta: {(delta_posterior > 0.5).astype(int)}")