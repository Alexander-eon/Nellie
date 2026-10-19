import numpy as np
import pandas as pd

def calc_te(x, y, bins=8):
    """
    Считает Transfer Entropy: TE(X -> Y).
    То есть, насколько прошлое X помогает предсказать будущее Y, 
    учитывая прошлое Y.
    """
    # Дискретизация непрерывных данных
    x_bin = np.digitize(x, np.linspace(np.min(x), np.max(x), bins))
    y_bin = np.digitize(y, np.linspace(np.min(y), np.max(y), bins))
    
    n = len(x) - 1
    if n < 10: return 0.0
    
    # Формируем пары (y_t, x_t, y_t+1)
    y_t = y_bin[:-1]
    x_t = x_bin[:-1]
    y_next = y_bin[1:]
    
    # Считаем совместные и маргинальные вероятности
    # Это упрощенная, но рабочая реализация для наших целей
    te = 0.0
    for i in range(n):
        p_yt_xt_yn = np.sum((y_t == y_t[i]) & (x_t == x_t[i]) & (y_next == y_next[i])) / n
        p_yt_yn = np.sum((y_t == y_t[i]) & (y_next == y_next[i])) / n
        p_yt_xt = np.sum((y_t == y_t[i]) & (x_t == x_t[i])) / n
        p_yt = np.sum(y_t == y_t[i]) / n
        
        if p_yt_xt_yn > 0 and p_yt > 0:
            # Формула TE
            p_cond1 = p_yt_xt_yn / p_yt_xt if p_yt_xt > 0 else 0
            p_cond2 = p_yt_yn / p_yt if p_yt > 0 else 0
            if p_cond1 > 0 and p_cond2 > 0:
                te += p_yt_xt_yn * np.log2(p_cond1 / p_cond2)
                
    return max(0.0, te)

def main():
    print("Загрузка данных...")
    df = pd.read_csv('comm_data.csv')
    
    print(f"Загружено {len(df)} тактов.\n")
    
    # Ключевые проверки:
    # 1. Влияет ли сообщение A на GPU (использует ли B сообщение A для контроля)
    te_msg_a_to_gpu = calc_te(df['msg_a'].values, df['gpu'].values)
    
    # 2. Влияет ли сообщение B на CPU (использует ли A сообщение B для контроля)
    te_msg_b_to_cpu = calc_te(df['msg_b'].values, df['cpu'].values)
    
    # 3. Кодирует ли A цель GPU в сообщении?
    te_targ_gpu_to_msg_a = calc_te(df['target_gpu'].values, df['msg_a'].values)
    
    # 4. Кодирует ли B цель CPU в сообщении?
    te_targ_cpu_to_msg_b = calc_te(df['target_cpu'].values, df['msg_b'].values)
    
    print("=== РЕЗУЛЬТАТЫ ТРАНСФЕРНОЙ ЭНТРОПИИ ===")
    print(f"TE(Цель_GPU -> msg_a): {te_targ_gpu_to_msg_a:.4f}  (A кодирует цель для B)")
    print(f"TE(Цель_CPU -> msg_b): {te_targ_cpu_to_msg_b:.4f}  (B кодирует цель для A)")
    print(f"TE(msg_a -> GPU):      {te_msg_a_to_gpu:.4f}  (B использует сообщение A)")
    print(f"TE(msg_b -> CPU):      {te_msg_b_to_cpu:.4f}  (A использует сообщение B)")
    print("=======================================")
    
    if te_msg_a_to_gpu > 0.01 and te_msg_b_to_cpu > 0.01:
        print("ВЕРДИКТ: ДВУСТОРОННЯЯ КОММУНИКАЦИЯ ПОДТВЕРЖДЕНА! 🤝")
    else:
        print("ВЕРДИКТ: Связь слабая, нужно больше данных или другие бины.")

if __name__ == "__main__":
    main()
