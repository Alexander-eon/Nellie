#!/bin/bash
# Последовательный запуск 5 прогонов с разными seed

for seed in 1 2 3 4 5; do
    echo "=== Запуск с seed=$seed ==="
    python3 evolution_sep.py \
        --generations 10 \
        --pairs 10 \
        --steps 2000 \
        --seed $seed \
        --log evolution_repro_seed${seed}.csv \
        > /tmp/evo_repro_seed${seed}.log 2>&1
    echo "seed=$seed завершён"
done

echo "Все 5 прогонов завершены"
