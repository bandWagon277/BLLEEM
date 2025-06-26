from BLLEEM_cpu import BLLEEMGibbsSampler
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict

def compute_y_mean(w,alpha,delta,X,Z,beta):
    m_jt = np.median(X, axis=0)
    X_centered = X - m_jt
    mean = np.sum(np.sum(w * X_centered, axis=2) * (alpha * delta), axis=1) + Z @ beta
    return mean

def main_BLLEEM(n,P,T,L,rho = 0.8,R2 = 0.2,enet = False):
    #delta = np.random.binomial(n=1, p=0.2, size=P)
    #delta = np.zeros(20, dtype=int)
    #indices = np.random.choice(20, size=5, replace=False)
    #delta[indices] = 1
    print(delta)
    file_path = r'/home/gengyh/BLLEM/save_delta.txt'
    with open(file_path, 'ab') as f:  # 'ab' 表示二进制追加模式
        np.savetxt(f, delta.reshape(1, -1), fmt='%d', delimiter=',')
    X = np.zeros((n, P, T))
    var_corr = 0.2
    """for p in range(P):
        for i in range(n):
            X[i, p, 0] = np.random.normal(0, 1) #initial value
            for t in range(1, T):
                X[i, p, t] = rho * X[i, p, t - 1] + np.random.normal(0, 1)"""

    Sigma = np.full((P, P), var_corr)
    np.fill_diagonal(Sigma, 1.0)

    # 对每个样本 i：
    for i in range(n):
        # 第一个时间点 t=0：变量之间相关，但无时间相关
        X[i, :, 0] = np.random.multivariate_normal(mean=np.zeros(P), cov=Sigma)

        # 后续时间点 t=1,...,T-1
        for t in range(1, T):
            eps_t = np.random.multivariate_normal(mean=np.zeros(P), cov=Sigma)
            X[i, :, t] = rho * X[i, :, t - 1] + eps_t
    Z = np.hstack([np.ones((n,1)), np.random.normal(size=(n, L))])  #include intercept

    mean = compute_y_mean(w,alpha,delta,X,Z,beta)
    #y = np.random.normal(loc = mean,scale = 1,size=n)
    var_mu = np.var(mean)
    sigma_epsilon = np.sqrt(var_mu * (1 - R2) / R2)

    # Generate response variable y
    y = mean + np.random.normal(0, sigma_epsilon, size=n)

    # Compute empirical R²
    R2_empirical = np.var(mean) / (np.var(mean) + sigma_epsilon**2)


    sampler_0 = BLLEEMGibbsSampler(y, X, Z, P, T, L, n_iter=2000) #fix w
    samples_0 = sampler_0.gibbs_sample()


    alpha_posterior_mean = 0
    beta_posterior_mean = 0
    delta_posterior_mean = 0
    w_posterior_mean = 0

    preserve = int(len(samples_0)/2)

    for i in range(len(samples_0)-preserve,len(samples_0)):

        alpha_posterior_mean += samples_0[i]["alpha"]
        beta_posterior_mean += samples_0[i]["beta"]
        delta_posterior_mean += samples_0[i]["delta"]
        w_posterior_mean += samples_0[i]["w"]


    #assume delta cutoff to be 0.5
    a = alpha_posterior_mean/preserve
    est_delta = (delta_posterior_mean/preserve > 0.5).astype(int)
    w_est = w_posterior_mean/preserve
    beta_est = beta_posterior_mean/preserve

    predict_y = compute_y_mean(w_est,a,est_delta,X,Z,beta_est)
    MSE = np.mean((y-predict_y)**2)

    para = (a*est_delta)[:, np.newaxis]*w_est
    importance_scores = np.abs(para)
    #consider direction
    #importance_dict = {"x_{}_{}".format(i,j): para[i,j] for i in range(P) for j in range(T)}
    flat_para = para.flatten()
    ranks = (-flat_para).argsort().argsort()  # 从小到大排名（从0开始）

    # 转换为字典：x_i_j -> rank（+1 让排名从 1 开始）
    importance_dict = {}
    for i in range(P):
        for j in range(T):
            idx = i * T + j
            importance_dict[f"x_{i}_{j}"] = int(ranks[idx]) + 1

    result={"Rank":importance_dict,"MSE":MSE}

    #consider direction
    #importance_dict = {"x_{}_{}".format(i,j): para[i,j] for i in range(P) for j in range(T)}

    #consider credible interval
    #t_stat = para / std_err
    #importance = np.abs(t_stat)

    return result


# 假设 results 是你的结果列表
# results = [{"Rank": {...}, "MSE": ...}, ...]

def summarize_results(results):
    rank_accumulator = defaultdict(list)
    mse_list = []

    for result in results:
        rank_dict = result["Rank"]
        mse_list.append(result["MSE"])

        for key, rank in rank_dict.items():
            rank_accumulator[key].append(rank)

    # 统计每个 key 的 mean 和 sd
    rank_summary = {}
    for key, ranks in rank_accumulator.items():
        ranks = np.array(ranks)
        rank_summary[key] = {
            "mean": ranks.mean(),
            "sd": ranks.std()
        }

    # 统计 MSE 的 mean 和 sd
    mse_array = np.array(mse_list)
    mse_summary = {
        "mean": mse_array.mean(),
        "sd": mse_array.std()
    }

    return rank_summary, mse_summary


if __name__ == "__main__":
    #predefine data
    n = 100
    P = 20
    L = 3
    T = 3

    #np.random.seed(123)
    #predefine para
    

    alpha = np.array([1.3,2,3.6,-0.9,1.1,2.5,-1.2,-3.1,2.4,-1.6,-1.7,3.9,-2.2,2.8,-4.1,1.9,0.8,-2.5,1.8,-2.9])
    #beta = np.random.normal(loc=0, scale=1, size=L+1)
    beta = np.array([-0.73521696, 0.50124899, 1.01273905, 0.27874086])
    """w = np.array([
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.8, 0.1, 0.1],
        [0.6, 0.1, 0.3],
        [0.7, 0.1, 0.2],
        [0.5, 0.3, 0.2],
        [5.99753932e-02, 8.01020196e-01, 1.39004411e-01],
        [4.07307441e-01, 2.19293425e-02, 5.70763217e-01],
        [6.83315799e-02, 1.01146265e-01, 8.30522155e-01],
        [2.40822327e-01, 1.99796029e-01, 5.59381644e-01],
        [0.2, 0.6, 0.2],
        [9.46413659e-01, 5.88439210e-04, 5.29979018e-02],
        [4.04282916e-02, 1.23693224e-01, 8.35878484e-01],
        [1.15689749e-01, 7.26531615e-01, 1.57778636e-01],
        [8.18148162e-01, 5.13710738e-02, 1.30480764e-01],
        [0.1, 0.2, 0.7],
        [4.68661942e-01, 4.91786264e-01, 3.95517938e-02],
        [6.01345884e-02, 2.35652579e-01, 7.04212832e-01],
        [3.88430998e-06, 9.54007029e-01, 4.59890870e-02],
        [0.333333, 0.333333, 0.333334]
    ])"""
    w = np.array([
    [0.150, 0.312, 0.538],  # increasing
    [0.530, 0.341, 0.129],  # decreasing
    [0.329, 0.333, 0.338],  # flat
    [0.135, 0.270, 0.595],  # increasing
    [0.314, 0.336, 0.350],  # flat
    [0.435, 0.360, 0.205],  # decreasing
    [0.339, 0.332, 0.329],  # flat
    [0.338, 0.328, 0.334],  # flat
    [0.619, 0.217, 0.164],  # decreasing
    [0.272, 0.358, 0.370],  # increasing
    [0.512, 0.354, 0.134],  # decreasing
    [0.333, 0.335, 0.332],  # flat
    [0.332, 0.300, 0.368],  # flat
    [0.144, 0.346, 0.510],  # increasing
    [0.132, 0.259, 0.609],  # increasing
    [0.582, 0.239, 0.179],  # decreasing
    [0.331, 0.335, 0.334],  # flat
    [0.613, 0.239, 0.148],  # decreasing
    [0.360, 0.328, 0.312],  # flat
    [0.151, 0.328, 0.521],  # increasing
    ])

    delta = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    #for true rank
    result_all = []
    for i in range(10):
        #delta = np.zeros(20, dtype=int)
        #indices = np.random.choice(20, size=5, replace=False)
        #delta[indices] = 1
        para = (alpha*delta)[:, np.newaxis]*w
        importance_scores = np.abs(para)
        #consider direction
        #importance_dict = {"x_{}_{}".format(i,j): para[i,j] for i in range(P) for j in range(T)}
        flat_para = para.flatten()
        ranks = (-flat_para).argsort().argsort()  # 从小到大排名（从0开始）

        # 转换为字典：x_i_j -> rank（+1 让排名从 1 开始）
        importance_dict = {}
        for i in range(P):
            for j in range(T):
                idx = i * T + j
                importance_dict[f"x_{i}_{j}"] = int(ranks[idx]) + 1

        MSE = 0

        result={"Rank":importance_dict,"MSE":MSE}
        result_all.append(result)

    rank_summary, mse_summary=summarize_results(result_all)
    print("MSE summary:", mse_summary)
    print("rank summaries:")
    for k in list(rank_summary.keys()):
        print(f"{k}: mean={rank_summary[k]['mean']:.2f}, sd={rank_summary[k]['sd']:.2f}")

    



    print("True alpha:",alpha)
    print("True beta:",beta)
    print("True w:",w)
    #print("True delta:",delta)
    
    result_1 = []
    n=500
    for i in range(10):
        result1 = main_BLLEEM(n,P,T,L,rho = 0.8,R2 = 0.2)
        result_1.append(result1)
        print(i)
    
    print("For blleem r=0.2:")
    rank_summary, mse_summary=summarize_results(result_1)
    print("MSE summary:", mse_summary)
    print("rank summaries:")
    for k in list(rank_summary.keys()):
        print(f"{k}: mean={rank_summary[k]['mean']:.2f}, sd={rank_summary[k]['sd']:.2f}")


    """result_1 = []
    n=300
    for i in range(10):
        result1 = main_BLLEEM(n,P,T,L,rho = 0.8,R2 = 0.5)
        result_1.append(result1)
        print(i)
    
    print("For blleem r=0.2:")
    point_statistic(result_1)"""
