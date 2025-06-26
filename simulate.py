from BLLEEM_cpu import BLLEEMGibbsSampler
import cupy as np
import torch
import multiprocessing as mp
from tqdm import tqdm

#import adelie as ad

def compute_y_mean(w,alpha,delta,X,Z,beta):
    m_jt = np.median(X, axis=0)
    X_centered = X - m_jt
    mean = np.sum(np.sum(w * X_centered, axis=2) * (alpha * delta), axis=1) + Z @ beta
    return mean


def main_sampling_cpu(n,P,T,L,alpha, delta, beta, w,rho = 0.8,R2 = 0.2):
    X = np.zeros((n, P, T))

    for p in range(P):
        for i in range(n):
            X[i, p, 0] = np.random.normal(0, 1) #initial value
            for t in range(1, T):
                X[i, p, t] = rho * X[i, p, t - 1] + np.random.normal(0, 1)
    Z = np.hstack([np.ones((n,1)), np.random.normal(size=(n, L))])  #include intercept

    mean = compute_y_mean(w,alpha,delta,X,Z,beta)
    #y = np.random.normal(loc = mean,scale = 1,size=n)
    var_mu = np.var(mean)
    sigma_epsilon = np.sqrt(var_mu * (1 - R2) / R2)

    # Generate response variable y
    y = mean + np.random.normal(0, sigma_epsilon, size=n)

    # Compute empirical R²
    R2_empirical = np.var(mean) / (np.var(mean) + sigma_epsilon**2)

    sampler_0 = BLLEEMGibbsSampler(y, X, Z, P, T, L, n_iter=20000) #fix w
    samples_0 = sampler_0.gibbs_sample()

    preserve = int(len(samples_0) / 10)
    samples_tail = samples_0[-preserve:]

    # 用 numpy 简洁地计算后验均值
    a = np.mean([s["alpha"] for s in samples_tail], axis=0)
    beta_posterior_mean = np.mean([s["beta"] for s in samples_tail], axis=0)
    delta_posterior_mean = np.mean([s["delta"] for s in samples_tail], axis=0)
    w_est = np.mean([s["w"] for s in samples_tail], axis=0)


    est_delta = (delta_posterior_mean > 0.5).astype(int)


    #delta statistics
    TP = np.sum(delta * est_delta)/np.sum(delta)
    FP = sum((est_delta == 1) & (delta == 0))/sum(delta == 0)
    FDR = sum((est_delta == 1) & (delta == 0))/ sum(est_delta==1) if sum(est_delta==1) > 0 else 0.0

    result["delta"] = {"TP":TP,"FP":FP,"FDR":FDR}


    #cate ==0
    delta_n = abs(delta-1)
    #alpha rmse
    subset = (np.array(alpha*delta_n) - np.array(a*delta_n)) ** 2
    a_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    #w mse
    true_w = delta_n[:, np.newaxis]*w
    est_w = delta_n[:, np.newaxis]*w_est
    subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    result["0"] = {"alpha error":a_rmse,"w error":w_rmse}


    #cate == 1
    #alpha rmse
    a_rmse = np.sqrt(np.mean((np.array(alpha*delta) - np.array(a*delta)) ** 2))
    #w mse
    true_w = delta[:, np.newaxis]*w
    est_w = delta[:, np.newaxis]*w_est
    subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    result["1"] = {"alpha error":a_rmse,"w error":w_rmse}

    
    #cate == 2
    #alpha rmse
    a_rmse = np.sqrt(np.mean((np.array(alpha*est_delta) - np.array(a*est_delta)) ** 2))
    #w mse
    true_w = est_delta[:, np.newaxis]*w
    est_w = est_delta[:, np.newaxis]*w_est
    subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    result["2"] = {"alpha error":a_rmse,"w error":w_rmse}

    return result



def compute_y_mean_torch(w, alpha, delta, X, Z, beta):
    # w: (P, T), alpha: (P,), delta: (P,), X: (n, P, T), Z: (n, L+1), beta: (L+1,)
    X_centered = X - X.median(dim=0).values  # (n, P, T)
    weighted_sum = torch.sum(torch.sum(w * X_centered, dim=2) * (alpha * delta), dim=1)
    return weighted_sum + Z @ beta

def main_sampling_torch(n, P, T, L, alpha, delta, beta, w, rho=0.8, R2=0.2, device='cuda'):
    device = torch.device('cpu') #device if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    # Simulate X on GPU
    X = torch.zeros((n, P, T), device=device)
    for p in range(P):
        X[:, p, 0] = torch.randn(n, device=device)
        for t in range(1, T):
            X[:, p, t] = rho * X[:, p, t - 1] + torch.randn(n, device=device)

    # Simulate Z on GPU (with intercept)
    Z = torch.cat([torch.ones(n, 1, device=device), torch.randn(n, L, device=device)], dim=1)

    # True mean and sigma
    mean = compute_y_mean_torch(w.to(device), alpha.to(device), delta.to(device), X, Z, beta.to(device))
    var_mu = torch.var(mean)
    sigma_epsilon = torch.sqrt(var_mu * (1 - R2) / R2)

    # Generate y
    y = mean + sigma_epsilon * torch.randn(n, device=device)

    # Initialize and run GPU Gibbs sampler
    sampler_0 = BLLEEMGibbsSampler(y.cpu().numpy(), X.cpu().numpy(), Z.cpu().numpy(), P, T, L, n_iter=2000) #GPU name
    samples_0 = sampler_0.gibbs_sample()

    preserve = int(len(samples_0) / 10)
    samples_tail = samples_0[-preserve:]

    # Post-processing: convert back to torch
    alpha_post = torch.tensor(np.mean([s["alpha"] for s in samples_tail], axis=0), device=device)
    beta_post = torch.tensor(np.mean([s["beta"] for s in samples_tail], axis=0), device=device)
    delta_post = torch.tensor(np.mean([s["delta"] for s in samples_tail], axis=0), device=device)
    w_post = torch.tensor(np.mean([s["w"] for s in samples_tail], axis=0), device=device)
    est_delta = (delta_post > 0.5).int()

    result = {}

    # Delta statistics
    TP = torch.sum(delta * est_delta) / torch.sum(delta)
    FP = torch.sum((est_delta == 1) & (delta == 0)) / torch.sum(delta == 0)
    FDR = torch.sum((est_delta == 1) & (delta == 0)) / torch.sum(est_delta == 1) if torch.sum(est_delta == 1) > 0 else torch.tensor(0.0, device=device)
    result["delta"] = {"TP": TP.item(), "FP": FP.item(), "FDR": FDR.item()}

    # Category 0: variables truly inactive
    delta_n = 1 - delta
    a_rmse = torch.sqrt(torch.mean(((alpha * delta_n - alpha_post * delta_n) ** 2))) if torch.sum(delta_n) > 0 else torch.tensor(0.0, device=device)
    mask = torch.any((delta_n[:, None] * w) != 0, dim=1)
    w_rmse = torch.sqrt(torch.mean((w[mask] - w_post[mask]) ** 2)) if torch.sum(mask) > 0 else torch.tensor(0.0, device=device)
    result["0"] = {"alpha error": a_rmse.item(), "w error": w_rmse.item()}

    # Category 1: variables truly active
    a_rmse = torch.sqrt(torch.mean(((alpha * delta - alpha_post * delta) ** 2)))
    mask = torch.any((delta[:, None] * w) != 0, dim=1)
    w_rmse = torch.sqrt(torch.mean((w[mask] - w_post[mask]) ** 2)) if torch.sum(mask) > 0 else torch.tensor(0.0, device=device)
    result["1"] = {"alpha error": a_rmse.item(), "w error": w_rmse.item()}

    # Category 2: variables estimated active
    a_rmse = torch.sqrt(torch.mean(((alpha * est_delta - alpha_post * est_delta) ** 2)))
    mask = torch.any((est_delta[:, None] * w) != 0, dim=1)
    w_rmse = torch.sqrt(torch.mean((w[mask] - w_post[mask]) ** 2)) if torch.sum(mask) > 0 else torch.tensor(0.0, device=device)
    result["2"] = {"alpha error": a_rmse.item(), "w error": w_rmse.item()}

    return result


def simulate_one_run(run_id, n, P, T, L, R2, rho, device='cuda'):
    torch.manual_seed(42 + run_id)

    """alpha = torch.randn(P, device=device)
    delta = (torch.rand(P, device=device) < 0.3).int()
    beta = torch.randn(L + 1, device=device)
    w = torch.rand(P, T, device=device)
    w = w / w.sum(dim=1, keepdim=True)"""

    alpha = np.array([1.3,2,3.6,-0.9,1.1,2.5,-1.2,-3.1,2.4,-1.6,-1.7,3.9,-2.2,2.8,-4.1,1.9,0.8,-2.5,1.8,-2.9])
    beta = np.array([-0.73521696, 0.50124899, 1.01273905, 0.27874086])
    w = np.array([
        [4.54652223e-01, 4.16434566e-02, 5.03704320e-01],
        [4.55749393e-01, 2.32058695e-01, 3.12191912e-01],
        [4.45233342e-02, 3.79217760e-01, 5.76258906e-01],
        [4.55273189e-01, 4.51131672e-01, 9.35951384e-02],
        [3.07666878e-01, 3.02550001e-02, 6.62078122e-01],
        [6.23022647e-02, 6.24548155e-02, 8.75242920e-01],
        [5.99753932e-02, 8.01020196e-01, 1.39004411e-01],
        [4.07307441e-01, 2.19293425e-02, 5.70763217e-01],
        [6.83315799e-02, 1.01146265e-01, 8.30522155e-01],
        [2.40822327e-01, 1.99796029e-01, 5.59381644e-01],
        [9.42262669e-01, 4.64556281e-02, 1.12817031e-02],
        [9.46413659e-01, 5.88439210e-04, 5.29979018e-02],
        [4.04282916e-02, 1.23693224e-01, 8.35878484e-01],
        [1.15689749e-01, 7.26531615e-01, 1.57778636e-01],
        [8.18148162e-01, 5.13710738e-02, 1.30480764e-01],
        [6.96359336e-01, 2.96446500e-02, 2.73996014e-01],
        [4.68661942e-01, 4.91786264e-01, 3.95517938e-02],
        [6.01345884e-02, 2.35652579e-01, 7.04212832e-01],
        [3.88430998e-06, 9.54007029e-01, 4.59890870e-02],
        [6.29067625e-01, 2.41523562e-02, 3.46780019e-01]
    ])
    delta = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    result = main_sampling_cpu(n, P, T, L, alpha, delta, beta, w, R2=R2, rho=rho)
    return result

def run_parallel_simulations(n_simulations=1000, n=100, P=20, T=3, L=3, R2=0.2, rho=0.8, device='cuda', n_jobs=4):
    # Prepare arguments for each process
    args = [(i, n, P, T, L, R2, rho, device) for i in range(n_simulations)]

    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.starmap(simulate_one_run, args), total=n_simulations))

    return results

if __name__ == "__main__":
    torch.set_num_threads(1)  


    """alpha = torch.tensor([1.3,2,3.6,-0.9,1.1,2.5,-1.2,-3.1,2.4,-1.6,-1.7,3.9,-2.2,2.8,-4.1,1.9,0.8,-2.5,1.8,-2.9])
    beta = torch.tensor([-0.73521696, 0.50124899, 1.01273905, 0.27874086])
    w = torch.tensor([
        [4.54652223e-01, 4.16434566e-02, 5.03704320e-01],
        [4.55749393e-01, 2.32058695e-01, 3.12191912e-01],
        [4.45233342e-02, 3.79217760e-01, 5.76258906e-01],
        [4.55273189e-01, 4.51131672e-01, 9.35951384e-02],
        [3.07666878e-01, 3.02550001e-02, 6.62078122e-01],
        [6.23022647e-02, 6.24548155e-02, 8.75242920e-01],
        [5.99753932e-02, 8.01020196e-01, 1.39004411e-01],
        [4.07307441e-01, 2.19293425e-02, 5.70763217e-01],
        [6.83315799e-02, 1.01146265e-01, 8.30522155e-01],
        [2.40822327e-01, 1.99796029e-01, 5.59381644e-01],
        [9.42262669e-01, 4.64556281e-02, 1.12817031e-02],
        [9.46413659e-01, 5.88439210e-04, 5.29979018e-02],
        [4.04282916e-02, 1.23693224e-01, 8.35878484e-01],
        [1.15689749e-01, 7.26531615e-01, 1.57778636e-01],
        [8.18148162e-01, 5.13710738e-02, 1.30480764e-01],
        [6.96359336e-01, 2.96446500e-02, 2.73996014e-01],
        [4.68661942e-01, 4.91786264e-01, 3.95517938e-02],
        [6.01345884e-02, 2.35652579e-01, 7.04212832e-01],
        [3.88430998e-06, 9.54007029e-01, 4.59890870e-02],
        [6.29067625e-01, 2.41523562e-02, 3.46780019e-01]
    ])
    delta = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    mp.set_start_method("spawn", force=True)

    results = run_parallel_simulations(
        n_simulations=10,
        n=100, P=20, T=3, L=3,
        R2=0.2, rho=0.8,
        device='cuda',
        n_jobs=4 
    )"""

    simulate_one_run(run_id = 0, n=100, P=20, T=3, L=3,R2=0.2, rho=0.8,device='cuda')

    print("First simulation", results[0])









    

