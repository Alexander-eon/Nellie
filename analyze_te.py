import numpy as np
import pandas as pd
from scipy.stats import entropy

def transfer_entropy(x, y, bins=10):
    """
    Правильный расчет Transfer Entropy (TE) через условную энтропию.
    TE(X -> Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-1)
    """
    # 1. Дискретизация (перевод непрерывных значений в категории)
    x_bin = np.digitize(x, np.linspace(np.min(x), np.max(x), bins))
    y_bin = np.digitize(y, np.linspace(np.min(y), np.max(y), bins))
    
    # 2. Формируем пары прошлое-будущее
    y_t = y_bin[:-1]
    x_t = x_bin[:-1]
    y_next = y_bin[1:]
    
    n = len(y_next)
    
    # 3. Считаем энтропию H(Y_t | Y_t-1)
    # Совместная энтропия H(Y_t-1, Y_t)
    joint_y_ynext = {}
    for i in range(n):
        pair = (y_t[i], y_next[i])
        joint_y_ynext[pair] = joint_y_ynext.get(pair, 0) + 1
        
    # Энтропия H(Y_t-1)
    y_t_counts = np.bincount(y_t, minlength=bins+1)
    h_y_t = entropy(y_t_counts, base=2)
    
    # H(Y_t | Y_t-1) = H(Y_t-1, Y_t) - H(Y_t-1)
    h_joint_y = entropy(list(joint_y_ynext.values()), base=2)
    h_y_given_y = h_joint_y - h_y_t
    
    # 4. Считаем энтропию H(Y_t | Y_t-1, X_t-1)
    joint_all = {}
    joint_x_y = {}
    x_counts = np.bincount(x_t, minlength=bins+1)
    
    for i in range(n):
        pair_xy = (x_t[i], y_t[i])
        joint_xy_yn = (x_t[i], y_t[i], y_next[i])
        joint_x_y[pair_xy] = joint_x_y.get(pair_xy, 0) + 1
        joint_all[joint_xy_yn] = joint_all.get(joint_xy_yn, 0) + 1
        
    h_joint_all = entropy(list(joint_all.values()), base=2)
    h_joint_xy = entropy(list(joint_x_y.values()), base=2)
    
    # H(Y_t | Y_t-1, X_t-1) = H(X, Y, Y_next) - H(X, Y)
    h_y_given_xy = h_joint_all - h_joint_xy
    
    # 5. TE = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-1)
    te = h_y_given_y - h_y_given_xy
    return max(0.0, te)

def main():
    df = pd.read_csv('comm_data.csv')
    print(f"Загружено {len(df)} тактов.\n")
    
    # Считаем TE
    te_tg_ma = transfer_entropy(df['target_gpu'].values, df['msg_a'].values)
    te_tc_mb = transfer_entropy(df['target_cpu'].values, df['msg_b'].values)
    te_ma_gpu = transfer_entropy(df['msg_a'].values, df['gpu'].values)
    te_mb_cpu = transfer_entropy(df['msg_b'].values, df['cpu'].values)
    
    print("=== РЕЗУЛЬТАТЫ (в битах) ===")
    print(f"TE(Цель_GPU -> msg_a): {te_tg_ma:.4f}  (A передает цель B)")
    print(f"TE(Цель_CPU -> msg_b): {te_tc_mb:.4f}  (B передает цель A)")
    print(f"TE(msg_a -> GPU):      {te_ma_gpu:.4f}  (B использует msg_a)")
    print(f"TE(msg_b -> CPU):      {te_mb_cpu:.4f}  (A использует msg_b)")
    print("============================")

if __name__ == "__main__":
    main()
