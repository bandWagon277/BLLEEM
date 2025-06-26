from BLLEEM_mix_cc import BLLEEMGibbsSampler
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


def compute_y_mean(w,alpha,delta,X,Z,beta):
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




def main_BLLEEM(n,P,T,L,rho = 0.8,R2 = 0.2,enet = False):
    #delta = np.random.binomial(n=1, p=0.2, size=P)
    """delta = np.zeros(20, dtype=int)
    indices = np.random.choice(20, size=5, replace=False)
    delta[indices] = 1
    print(delta)"""
    delta = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
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
    p_posterior_mean = 0

    preserve = int(len(samples_0)/2)

    for i in range(len(samples_0)-preserve,len(samples_0)):

        alpha_posterior_mean += samples_0[i]["alpha"]
        beta_posterior_mean += samples_0[i]["beta"]
        delta_posterior_mean += samples_0[i]["delta"]
        w_posterior_mean += samples_0[i]["w"]
        #p_posterior_mean += samples_0[i]["p"]


    result = {"0":{},"1":{},"2":{},"delta":{},"cate":{}}
    #assume delta cutoff to be 0.5
    a = alpha_posterior_mean/preserve
    est_delta = (delta_posterior_mean/preserve > 0.5).astype(int)
    w_est = w_posterior_mean/preserve
    #print("Estimate p",p_posterior_mean/preserve)

    #delta statistics
    TP = np.sum(delta * est_delta)/(np.sum(delta)+1e-6)
    FP = sum((est_delta == 1) & (delta == 0))/sum(delta == 0)
    FDR = sum((est_delta == 1) & (delta == 0))/ sum(est_delta==1) if sum(est_delta==1) > 0 else 0.0

    result["delta"] = {"TP":TP,"FP":FP,"FDR":FDR}

    #cate ==0
    """delta_n = abs(delta-1)
    #alpha rmse
    subset = (np.array(alpha*delta_n) - np.array(a*delta_n)) ** 2
    a_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    #w mse
    true_w = delta_n[:, np.newaxis]*w
    est_w = delta_n[:, np.newaxis]*w_est
    subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    result["0"] = {"alpha error":a_rmse,"w error":w_rmse}"""
    mask = delta == 0
    #alpha rmse
    subset = (np.zeros(P)[mask] - np.array((a*est_delta)[mask])) ** 2
    a_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    #w mse
    true_w = np.zeros((P,T))
    est_w = est_delta[:, np.newaxis]*w_est
    w_rmse = np.sqrt(np.mean((true_w[mask] - est_w[mask])**2) )
    result["0"] = {"alpha error":a_rmse,"w error":w_rmse}



    #cate == 1
    """#alpha rmse
    a_rmse = np.sqrt(np.mean((np.array(alpha*delta) - np.array(a*delta)) ** 2))
    #w mse
    true_w = delta[:, np.newaxis]*w
    est_w = delta[:, np.newaxis]*w_est
    subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    result["1"] = {"alpha error":a_rmse,"w error":w_rmse}"""
    mask = delta == 1
    #alpha rmse
    a_rmse = np.sqrt(np.mean((np.array((alpha*delta)[mask]) - np.array((a*est_delta)[mask])) ** 2))
    #w mse
    #true_w = delta[:, np.newaxis]*w
    true_w = w
    est_w = est_delta[:, np.newaxis]*w_est
    #subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean((true_w[mask] - est_w[mask])**2))
    result["1"] = {"alpha error":a_rmse,"w error":w_rmse}

    
    #cate == 2
    """#alpha rmse
    a_rmse = np.sqrt(np.mean((np.array(alpha*est_delta) - np.array(a*est_delta)) ** 2))
    #w mse
    true_w = est_delta[:, np.newaxis]*w
    est_w = est_delta[:, np.newaxis]*w_est
    subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean(subset) if subset.size > 0 else 0)
    result["2"] = {"alpha error":a_rmse,"w error":w_rmse}"""
    #alpha rmse
    a_rmse = np.sqrt(np.mean((np.array(alpha*delta) - np.array(a*est_delta)) ** 2))
    #w mse
    true_w = delta[:, np.newaxis]*w
    est_w = est_delta[:, np.newaxis]*w_est
    #subset = (np.array(true_w[~np.all(true_w == 0, axis=1)]) - np.array(est_w[~np.all(est_w == 0, axis=1)])) ** 2
    w_rmse = np.sqrt(np.mean((true_w - est_w)**2))
    #print(w_rmse)
    result["2"] = {"alpha error":a_rmse,"w error":w_rmse}


    #category acc
    acc  = per_class_category_accuracy(true_w, est_w)
    result["cate"] = acc

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
    [0.0, 0.46, 0.54],  # increasing
    [0.530, 0.341, 0.129],  # decreasing
    [0.329, 0.333, 0.338],  # flat
    [0.0, 0.33, 0.67],  # increasing
    [0.314, 0.336, 0.350],  # flat
    [0.435, 0.360, 0.205],  # decreasing
    [0.339, 0.332, 0.329],  # flat
    [0.338, 0.328, 0.334],  # flat
    [0.62, 0.38, 0.0],  # decreasing
    [0.0, 0.22, 0.78],  # increasing
    [0.56, 0.44, 0.0],  # decreasing
    [0.333, 0.335, 0.332],  # flat
    [0.332, 0.300, 0.368],  # flat
    [0.144, 0.346, 0.510],  # increasing
    [0.132, 0.259, 0.609],  # increasing
    [0.85, 0.15, 0.0],  # decreasing
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
    n=300
    for i in range(3):
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


