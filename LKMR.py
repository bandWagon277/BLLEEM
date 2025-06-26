from BLLEEM_mix_cc import BLLEEMGibbsSampler
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import multivariate_normal, norm
from sklearn.preprocessing import StandardScaler

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
def simulate_exposure_data(n=100, num_time=4, grp_size=5, ar_coef=0.8, seed=1):
    """
    生成混合暴露时间序列模拟数据
    
    Parameters:
    -----------
    n : int, 样本量
    num_time : int, 时间点数量
    grp_size : int, 每个时间点的暴露变量数量
    ar_coef : float, 时间序列自相关系数
    seed : int, 随机种子
    
    Returns:
    --------
    dict: 包含所有模拟数据的字典
    """
    
    np.random.seed(seed)
    
    # 1. 构建时间自相关矩阵 (AR-1结构)
    def autocorr_matrix(p, rho):
        """构建AR(1)自相关矩阵"""
        indices = np.arange(p)
        return rho ** np.abs(indices[:, None] - indices[None, :])
    
    time_mat = autocorr_matrix(num_time, ar_coef)
    
    # 2. 金属间相关矩阵 (基于真实金属污染物相关性)
    metal_mat = np.array([
        [1.00, 0.19, 0.32, 0.15, 0.40],
        [0.19, 1.00, 0.11, 0.25, 0.40],
        [0.32, 0.11, 1.00, 0.17, 0.24],
        [0.15, 0.25, 0.17, 1.00, 0.35],
        [0.40, 0.40, 0.24, 0.35, 1.00]
    ])
    
    # 3. 构建总协方差矩阵 (Kronecker积)
    cov_matrix = np.kron(time_mat, metal_mat)
    
    # 4. 生成暴露数据
    mean_exposure = np.zeros(grp_size * num_time)
    Z = multivariate_normal.rvs(mean=mean_exposure, cov=cov_matrix, size=n)
    
    # 将暴露数据按时间点分组
    Z_dict = {}
    for t in range(num_time):
        start_idx = t * grp_size
        end_idx = (t + 1) * grp_size
        Z_dict[f'Z_{t+1}'] = Z[:, start_idx:end_idx]
    
    # 5. Generate covariates (three continuous covariates + intercept)
    U1 = norm.rvs(loc=10, scale=1, size=n)
    U2 = norm.rvs(loc=5, scale=2, size=n)
    U3 = norm.rvs(loc=0, scale=1, size=n)

    # Standardize U1–U3
    scaler = StandardScaler()
    U_all = np.column_stack([U1, U2, U3])
    U_standardized = scaler.fit_transform(U_all)

    # Add intercept (column of ones)
    intercept = np.ones((n, 1))
    U = np.hstack([intercept, U_standardized])  # shape: (n, 4)

    # 6. Covariate effects
    conf_effects = np.array([1.0, 1.0, 0.5, -0.5])  # intercept + U1, U2, U3 coefficients
    covariate_effect = U @ conf_effects

    
    # 线性暴露效应函数（去除交互项和高次项）
    def linear_exposure_effect(Z_t, coef):
        """计算线性暴露效应（仅z1和z2有效应）"""
        z1, z2 = Z_t[:, 0], Z_t[:, 1]
        return coef * (z1 + z2)  # 简单线性效应，z3,z4,z5为噪声
    
    # 计算各时间点的暴露效应
    exposure_effects = np.zeros(n)
    
    #exposure_effects += 0
    exposure_effects += linear_exposure_effect(Z_dict['Z_2'], 0.5)
    exposure_effects += linear_exposure_effect(Z_dict['Z_3'], 0.8)
    exposure_effects += linear_exposure_effect(Z_dict['Z_4'], 1.0)
    
    error_term = norm.rvs(loc=0, scale=1, size=n)
    
    # 最终响应变量
    Y = covariate_effect + exposure_effects + error_term
    
    # 7. 计算真实效应函数值
    true_effects = {}
    true_effects['time_1'] = np.zeros(n)  # 时间点1无效应
    true_effects['time_2'] = linear_exposure_effect(Z_dict['Z_2'], 0.5)
    true_effects['time_3'] = linear_exposure_effect(Z_dict['Z_3'], 0.8) 
    true_effects['time_4'] = linear_exposure_effect(Z_dict['Z_4'], 1.0)
    
    return {
        'Y': Y,
        'Z': Z,
        'Z_by_time': Z_dict,
        'U': U,
        'true_effects': true_effects,
        'covariate_effect': covariate_effect,
        'exposure_effects': exposure_effects,
        'time_correlation': time_mat,
        'metal_correlation': metal_mat,
        'total_correlation': cov_matrix,
        'parameters': {
            'n': n,
            'num_time': num_time,
            'grp_size': grp_size,
            'ar_coef': ar_coef,
            'effect_coefs': [0, 0.5, 0.8, 1.0]
        }
    }

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
    delta = np.array([1,1,0,0,0])

    alpha = np.array([2.3,2.3,0,0,0])
    w = np.array([
    [0,0.217, 0.348, 0.435],
    [0,0.217, 0.348, 0.435],
    [0,0.217, 0.348, 0.435],
    [0,0.217, 0.348, 0.435],
    [0,0.217, 0.348, 0.435],
    ])

    data = simulate_exposure_data(n)
    Z_dict = data['Z_by_time']
    X = np.stack([Z_dict[f'Z_{t+1}'] for t in range(T)], axis=2)  # shape: (n, grp_size, num_time)

    sampler_0 = BLLEEMGibbsSampler(data['Y'], X, data['U'], P, T, L, n_iter=2000) #fix w
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


    result = {"0":{},"1":{},"2":{},"delta":{},"cate":{}}
    #assume delta cutoff to be 0.5
    a = alpha_posterior_mean/preserve
    est_delta = (delta_posterior_mean/preserve > 0.5).astype(int)
    w_est = w_posterior_mean/preserve


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
    n = 1000
    P = 5
    L = 3
    T = 4
    
    result_1 = []
    #n=100
    for i in range(3):
        result1 = main_BLLEEM(n,P,T,L,rho = 0.8,R2 = 0.5)
        result_1.append(result1)
        print(i)
    
    print("For blleem r=0.2:")
    point_statistic(result_1)