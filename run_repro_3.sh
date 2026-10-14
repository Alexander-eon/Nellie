#!/bin/bash
for seed in 1 2 3; do
    echo "=== Запуск с seed=$seed ==="
    python3 evolution_sep_disc_msg.py \
        --generations 10 \
        --pairs 10 \
        --steps 2000 \
        --seed $seed \
        > /tmp/nt_repro_seed${seed}.log 2>&1
    echo "seed=$seed завершён"
done
echo "Все 3 прогона завершены"
