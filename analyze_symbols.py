import pandas as pd

def main():
    df = pd.read_csv('symbols_data.csv')
    
    print("=== АНАЛИЗ СИМВОЛИЧЕСКОГО ЯЗЫКА ===\n")
    
    # 1. Сколько всего раз использовались символы
    total_steps = len(df)
    count_00 = len(df[df['sym_a'] == 0.0]) + len(df[df['sym_b'] == 0.0])
    count_10 = len(df[df['sym_a'] == 1.0]) + len(df[df['sym_b'] == 1.0])
    count_05 = len(df[df['sym_a'] == 0.5]) + len(df[df['sym_b'] == 0.5])
    
    print("--- 1. Частота использования символов ---")
    print(f"Символ 0.0 (опасность): {count_00} раз")
    print(f"Символ 1.0 (рост):      {count_10} раз")
    print(f"Символ 0.5 (стабильно): {count_05} раз")
    
    # 2. Проверяем, совпадают ли шоки с символами
    print("\n--- 2. Проверка на шоках ---")
    
    shock_cpu_idx = df[df['event'] == 'SHOCK_CPU'].index.tolist()
    shock_gpu_idx = df[df['event'] == 'SHOCK_GPU'].index.tolist()
    
    for name, indices, agent_col in [("SHOCK_CPU", shock_cpu_idx, 'sym_a'), ("SHOCK_GPU", shock_gpu_idx, 'sym_b')]:
        if len(indices) == 0:
            print(f"{name}: шоков не было.")
            continue
            
        success = 0
        for idx in indices:
            # Смотрим на 1 такт ДО шока и 2 такта ПОСЛЕ
            window = df.iloc[max(0, idx-1):idx+3]
            if 0.0 in window[agent_col].values or 1.0 in window[agent_col].values:
                success += 1
                
        print(f"{name}: {success} из {len(indices)} шоков сопровождались сменой символа (0.0 или 1.0)")
        
    # 3. Визуализация первых 200 шагов (для понимания)
    print("\n--- 3. Примеры в логе (первые 200 шагов) ---")
    sample = df.head(200)
    changed = sample[sample['sym_a'] != 0.5]
    print(f"Шаги, где Агент А передал 0.0 или 1.0: {changed['step'].tolist()}")
    changed_b = sample[sample['sym_b'] != 0.5]
    print(f"Шаги, где Агент B передал 0.0 или 1.0: {changed_b['step'].tolist()}")

if __name__ == "__main__":
    main()
