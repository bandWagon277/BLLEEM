from BLLEEM_binary import BLLEEMBinaryGibbsSampler
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict

"""# 用于包装 tqdm 的 helper 函数
def run_once_with_tqdm(args):
    i, queue = args
    result = main_BLLEEM(n, P, T, L, rho=0.8, R2=0.2)
    queue.put(1)  # 每完成一个就放入队列，触发进度更新
    return result"""

# 全局变量，用于子进程访问
q = None

# 子进程初始化函数：把 queue 绑定为全局变量
def init_worker(queue):
    global q
    q = queue

# 真正的工作函数（用到全局变量 q）
def run_once_with_tqdm(i):
    result = main_BLLEEM(n, P, T, L, rho=0.8, R2=0.2)
    q.put(1)  # 用全局变量发送进度
    return result


def run_once(_):
    return main_BLLEEM(n, P, T, L, rho=0.8, R2=0.2)


def compute_y_mean_binary(w, alpha, delta, X, Z, beta):
    """Compute mean of latent variable y_star"""
    m_jt = np.median(X, axis=0)
    X_centered = X - m_jt
    mean = np.sum(np.sum(w * X_centered, axis=2) * (alpha * delta), axis=1) + Z @ beta
    return mean

# sample multiple datasets


def point_statistic(result1):

    # 假设 result1 是一个长度为 1000 的 list，每个元素是一个 result dict
    # 示例：result1 = [result1_0, result1_1, ..., result1_999]
    # 初始化空字典来存放结果
    summary = {
        "0_alpha_error": [],
        "0_w_error": [],
        "1_alpha_error": [],
        "1_w_error": [],
        "2_alpha_error": [],
        "2_w_error": [],
        "delta_TP": [],
        "delta_FP": [],
        "delta_FDR": [],
        "cate_inc":[],
        "cate_dec":[],
        "cate_flat":[]
    }

    # 遍历每个 result，提取相应值
    for res in result1:
        summary["0_alpha_error"].append(res["0"]["alpha error"])
        summary["0_w_error"].append(res["0"]["w error"])
        summary["1_alpha_error"].append(res["1"]["alpha error"])
        summary["1_w_error"].append(res["1"]["w error"])
        summary["2_alpha_error"].append(res["2"]["alpha error"])
        summary["2_w_error"].append(res["2"]["w error"])
        summary["delta_TP"].append(res["delta"]["TP"])
        summary["delta_FP"].append(res["delta"]["FP"])
        summary["delta_FDR"].append(res["delta"]["FDR"])
        summary["cate_inc"].append(res["cate"]["increasing"])
        summary["cate_dec"].append(res["cate"]["decreasing"])
        summary["cate_flat"].append(res["cate"]["flat"])


    for k in summary:
        arr = np.array(summary[k])
        
        try:
            mean_val = np.nanmean(arr) #np.mean(arr)
            print(f"mean of {k}:", mean_val)
        except Exception as e:
            print(f"Error computing mean of {k} with value {arr}: {e}")
        
        try:
            std_val = np.nanstd(arr)#np.std(arr)
            print(f"std of {k}:", std_val)
        except Exception as e:
            print(f"Error computing std of {k} with value {arr}: {e}")


def get_category(w_row, threshold=0.07):
    diffs = np.diff(w_row)
    if np.all(diffs > threshold):
        return "increasing"
    elif np.all(diffs < -threshold):
        return "decreasing"
    else:
        return "flat"

def per_class_category_accuracy(w_true, w_est, threshold=0.07):
    assert w_true.shape == w_est.shape, "Shape mismatch"

    correct_dict = defaultdict(int)
    total_dict = defaultdict(int)

    for i in range(w_true.shape[0]):
        if np.all(w_true[i] == 0) or np.all(w_est[i] == 0):
            continue  # skip zero rows

        cat_true = get_category(w_true[i], threshold)
        cat_est  = get_category(w_est[i], threshold)

        total_dict[cat_true] += 1
        if cat_true == cat_est:
            correct_dict[cat_true] += 1

    # Compute accuracy per class
    categories = ["increasing", "decreasing", "flat"]
    acc_per_class = {}
    for cat in categories:
        if total_dict[cat] > 0:
            acc_per_class[cat] = correct_dict[cat] / total_dict[cat]
        else:
            acc_per_class[cat] = np.nan

    return acc_per_class


def main_BLLEEM(n, P, T, L, rho=0.8, R2=0.2, enet=False):
    """
    Main function for BLLEEM binary outcome simulation and sampling
    
    Parameters:
    n: sample size
    P: number of exposure variables  
    T: number of time points
    L: number of confounders (excluding intercept)
    rho: temporal correlation coefficient
    R2: desired R-squared for latent variable
    """
    
    # Generate true delta (variable selection indicator)
    delta = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    """delta = np.zeros(P, dtype=int)
    # Randomly select 5 variables to be active (modify as needed)
    n_active = min(5, P)  # Ensure we don't select more than P variables
    indices = np.random.choice(P, size=n_active, replace=False)
    delta[indices] = 1"""
    print(f"True delta: {delta}")
    
    # Save delta for record keeping
    """file_path = r'/home/gengyh/BLLEM/save_delta_binary.txt'
    try:
        with open(file_path, 'ab') as f:
            np.savetxt(f, delta.reshape(1, -1), fmt='%d', delimiter=',')
    except:
        print("Could not save delta to file")"""
    
    # Generate exposure data X with temporal and cross-sectional correlation
    X = np.zeros((n, P, T))
    var_corr = 0.2  # Cross-sectional correlation
    
    # Create correlation matrix for exposures
    Sigma = np.full((P, P), var_corr)
    np.fill_diagonal(Sigma, 1.0)
    
    # Generate correlated exposure data with temporal dependence
    for i in range(n):
        # First time point: variables correlated, no temporal correlation
        X[i, :, 0] = np.random.multivariate_normal(mean=np.zeros(P), cov=Sigma)
        
        # Subsequent time points with temporal correlation
        for t in range(1, T):
            eps_t = np.random.multivariate_normal(mean=np.zeros(P), cov=Sigma)
            X[i, :, t] = rho * X[i, :, t - 1] + eps_t
    
    # Generate confounder matrix (including intercept)
    Z = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, L))])
    
    # Set true parameters for simulation
    # You need to define these true parameters - here are example values
    alpha = np.random.normal(0, 1, P)  # True alpha coefficients
    alpha[delta == 0] = 0  # Set non-selected variables to 0
    
    # True weights (you can customize these)
    w = np.random.dirichlet(np.ones(T), size=P)  # Each row sums to 1
    
    # True beta coefficients  
    beta = np.random.normal(0, 1, L + 1)  # Including intercept
    
    print(f"True alpha: {alpha}")
    print(f"True beta: {beta}")
    print(f"True w shape: {w.shape}")
    
    # Compute latent variable mean
    mean = compute_y_mean_binary(w, alpha, delta, X, Z, beta)
    
    """# Adjust variance to achieve desired R2
    var_mu = np.var(mean)
    sigma_epsilon = np.sqrt(var_mu * (1 - R2) / R2)
    
    # Generate latent variable y_star
    y_star = mean + np.random.normal(0, sigma_epsilon, size=n)
    
    # Generate binary outcome
    y_binary = (y_star > 0).astype(int)
    
    # Compute empirical R² for latent variable
    R2_empirical = np.var(mean) / (np.var(mean) + sigma_epsilon**2)
    
    print(f"Target R²: {R2}, Empirical R²: {R2_empirical:.4f}")
    print(f"Binary outcome distribution: {np.bincount(y_binary)}")
    print(f"Proportion y=1: {np.mean(y_binary):.4f}")"""

    # Adjust variance to achieve desired R2 (probit-specific version)
    var_mu = np.var(mean)
    target_var_mu = R2 / (1 - R2)
    scaling_factor = np.sqrt(target_var_mu / var_mu)
    mean_scaled = mean * scaling_factor

    # Probit error variance is fixed to 1
    y_star = mean_scaled + np.random.normal(0, 1, size=n)
    y_binary = (y_star > 0).astype(int)

    # Check empirical probit R²
    R2_empirical = np.var(mean_scaled) / (np.var(mean_scaled) + 1)
    
    print(f"Target R²: {R2}, Empirical R²: {R2_empirical:.4f}")
    print(f"Binary outcome distribution: {np.bincount(y_binary)}")
    print(f"Proportion y=1: {np.mean(y_binary):.4f}")
    
    # Fit the binary BLLEEM model
    sampler_0 = BLLEEMBinaryGibbsSampler(y_binary, X, Z, P, T, L, n_iter=2000)
    samples_0 = sampler_0.gibbs_sample()
    
    # Compute posterior means (burn-in: discard first half)
    preserve = int(len(samples_0) / 2)
    
    alpha_posterior_mean = np.zeros(P)
    beta_posterior_mean = np.zeros(L + 1)
    delta_posterior_mean = np.zeros(P)
    w_posterior_mean = np.zeros((P, T))
    
    for i in range(len(samples_0) - preserve, len(samples_0)):
        alpha_posterior_mean += samples_0[i]["alpha"]
        beta_posterior_mean += samples_0[i]["beta"] 
        delta_posterior_mean += samples_0[i]["delta"]
        w_posterior_mean += samples_0[i]["w"]
    
    # Average over post-burn-in samples
    a = alpha_posterior_mean / preserve
    beta_est = beta_posterior_mean / preserve
    est_delta = (delta_posterior_mean / preserve > 0.5).astype(int)
    w_est = w_posterior_mean / preserve
    
    print(f"Estimated delta: {est_delta}")
    print(f"Estimated alpha: {a}")
    
    # Initialize results dictionary
    result = {"0": {}, "1": {}, "2": {}, "delta": {}, "cate": {}}
    
    # Delta (variable selection) statistics
    TP = np.sum(delta * est_delta) / (np.sum(delta) + 1e-6)  # True Positive Rate
    FP = np.sum((est_delta == 1) & (delta == 0)) / max(np.sum(delta == 0), 1)  # False Positive Rate
    FDR = np.sum((est_delta == 1) & (delta == 0)) / max(np.sum(est_delta == 1), 1)  # False Discovery Rate
    
    result["delta"] = {"TP": TP, "FP": FP, "FDR": FDR}
    
    # Category 0: Variables that should be inactive (delta == 0)
    mask = delta == 0
    if np.any(mask):
        # Alpha error: estimated alpha should be 0 for inactive variables
        subset = (np.zeros(P)[mask] - np.array((a * est_delta)[mask])) ** 2
        a_rmse = np.sqrt(np.mean(subset))
        
        # W error: estimated weights should be uniform for inactive variables
        true_w_inactive = np.ones((np.sum(mask), T)) / T  # Uniform weights
        est_w_inactive = w_est[mask]
        w_rmse = np.sqrt(np.mean((true_w_inactive - est_w_inactive) ** 2))
        
        result["0"] = {"alpha error": a_rmse, "w error": w_rmse}
    else:
        result["0"] = {"alpha error": 0.0, "w error": 0.0}
    
    # Category 1: Variables that should be active (delta == 1)
    mask = delta == 1
    if np.any(mask):
        # Alpha error for truly active variables
        a_rmse = np.sqrt(np.mean((np.array((alpha * delta)[mask]) - 
                                np.array((a * est_delta)[mask])) ** 2))
        
        # W error for truly active variables
        true_w_active = w[mask]
        est_w_active = w_est[mask]
        w_rmse = np.sqrt(np.mean((true_w_active - est_w_active) ** 2))
        
        result["1"] = {"alpha error": a_rmse, "w error": w_rmse}
    else:
        result["1"] = {"alpha error": 0.0, "w error": 0.0}
    
    # Category 2: Overall error (all variables)
    # Alpha error: compare true active effects vs estimated active effects
    a_rmse = np.sqrt(np.mean((np.array(alpha * delta) - np.array(a * est_delta)) ** 2))
    
    # W error: compare true weights vs estimated weights (weighted by selection)
    true_w_weighted = delta[:, np.newaxis] * w
    est_w_weighted = est_delta[:, np.newaxis] * w_est
    w_rmse = np.sqrt(np.mean((true_w_weighted - est_w_weighted) ** 2))
    
    result["2"] = {"alpha error": a_rmse, "w error": w_rmse}
    
    # Category accuracy (if you have this function defined)
    try:
        acc = per_class_category_accuracy(delta[:, np.newaxis] * w, 
                                        est_delta[:, np.newaxis] * w_est)
        result["cate"] = acc
    except NameError:
        # If per_class_category_accuracy is not defined, compute a simple accuracy
        correct_selection = np.mean(delta == est_delta)
        result["cate"] = {"overall_accuracy": correct_selection}
    
    # Additional metrics specific to binary outcomes
    result["binary_metrics"] = {
        "proportion_positive": np.mean(y_binary),
        "latent_R2": R2_empirical
        #"sigma_epsilon": sigma_epsilon
    }
    
    return result


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
    """w = np.array([
        # 1>2>3 (Category 1)
        [0.5, 0.3, 0.2],
        [0.7, 0.2, 0.1],
        
        # 1>3>2 (Category 2)
        [0.55, 0.15, 0.3],
        [0.6, 0.1, 0.3],
        
        # 1>2=3 (Category 3)
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1],
        
        # 2>1>3 (Category 4)
        [0.2, 0.65, 0.15],
        [0.25, 0.6, 0.15],
        
        # 2>3>1 (Category 5)
        [0.2, 0.5, 0.3],
        [0.1, 0.7, 0.2],
        
        # 2>3=1 (Category 6: b > c = a)
        [0.2, 0.6, 0.2],
        [0.25, 0.5, 0.25],
        
        # 3>2>1 (Category 7)
        [0.25, 0.35, 0.4],
        [0.1, 0.2, 0.7],
        
        # 3>1>2 (Category 8)
        [0.3, 0.2, 0.5],
        [0.4, 0.1, 0.5],
        
        # 3>1=2 (Category 9)
        [0.25, 0.25, 0.5],
        [0.2, 0.2, 0.6],
        
        # 3=1=2 (Category 10)
        [1/3, 1/3, 1/3],
        [1/3, 1/3, 1/3],
    ])"""
    #delta = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1])



    print("True alpha:",alpha)
    print("True beta:",beta)
    print("True w:",w)
    #print("True delta:",delta)
    
    result_1 = []
    n=100
    for i in range(12):
        result1 = main_BLLEEM(n,P,T,L,rho = 0.8,R2 = 0.5)
        result_1.append(result1)
        print(i)
    
    print("For blleem r=0.2:")
    point_statistic(result_1)


    """result_1 = []
    n=300
    for i in range(10):
        result1 = main_BLLEEM(n,P,T,L,rho = 0.8,R2 = 0.2)
        result_1.append(result1)
        print(i)
    
    print("For blleem r=0.2:")
    point_statistic(result_1)"""


    """num_runs = 10
    num_workers = mp.cpu_count()
    
    queue = mp.Queue()

    # 启动进度条监控
    def update_progress(q, total):
        with tqdm(total=total, desc="Running main_BLLEEM") as pbar:
            for _ in range(total):
                q.get()
                pbar.update(1)

    watcher = mp.Process(target=update_progress, args=(queue, num_runs))
    watcher.start()

    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(queue,)) as pool:
        result_1 = pool.map(run_once_with_tqdm, range(num_runs))

    watcher.join()

    print("For blleem:")
    point_statistic(result_1)

    print("For blleem:")
    point_statistic(result_1)"""


