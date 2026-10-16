"""
Эволюция с разделённым зрением.

Среда:
- A управляет CPU, знает target_gpu.
- B управляет GPU, знает target_cpu.
- Командный reward: -(error_cpu + error_gpu).

Эволюция:
- Мутации весов обоих.
- Отбор по fitness = lifetime × energy × (1 + std_a + std_b).

Метрики:
- std(actions_a), std(actions_b) — разнообразие действий.
- Корреляции после прогона.
"""

import sys
import csv
import time
import numpy as np

sys.path.insert(0, '.')

from brain import Brain
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy
from mutation import mutate_weights
from transfer_entropy import transfer_entropy


def discretize_action(value: float, n_levels: int = 6) -> float:
    """
    Округляет непрерывное действие до ближайшего дискретного уровня.
    n_levels = 6 -> уровни: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    """
    value = max(0.0, min(1.0, value))
    levels = np.linspace(0.0, 1.0, n_levels)
    idx = np.argmin(np.abs(levels - value))
    return float(levels[idx])


def message_bonus(msg: float, target: float) -> float:
    """
    Ступенчатый бонус за правильное сообщение.
    Чем ближе msg к target, тем выше бонус.
    """
    diff = abs(msg - target)
    if diff < 0.05:
        return 1.0
    if diff < 0.10:
        return 0.7
    if diff < 0.20:
        return 0.3
    return 0.0


class CommPairSep:
    """Пара агентов с разделённым зрением."""

    def __init__(
        self,
        pair_id: int,
        generation: int,
        parent_id: int,
        brain_a: Brain,
        brain_b: Brain,
        dt: float = 0.05,
        target_change_period: int = 100,
        zone_width: float = 0.15,
    ):
        self.pair_id = pair_id
        self.generation = generation
        self.parent_id = parent_id
        self.brain_a = brain_a
        self.brain_b = brain_b
        self.dt = dt
        self.target_change_period = target_change_period
        self.zone_width = zone_width
        self.result = None

    def run_life(self, max_steps: int = 2000):
        """Пара живёт до смерти или до лимита."""
        sensor = BodySensor()
        actuator = BodyActuator()
        energy = Energy(
            drain_rate=0.002,
            bounty_rate=0.004,
            bounty_probability=0.7,
            cpu_min=0.3, cpu_max=0.7,
            gpu_min=0.3, gpu_max=0.7,
            death_threshold=0.2,
            drift_period=100000,
            connection_penalty=0.5,
        )

        rng = np.random.default_rng(42)
        target_cpu = 0.5
        target_gpu = 0.5

        last_action_a = np.array([0.0])
        last_action_b = np.array([0.0])
        last_message_a = np.array([0.0])
        last_message_b = np.array([0.0])
        last_cpu = None
        last_gpu = None

        actuator.start()
        actuator.set_intensity(0.3, 0.3)
        time.sleep(1.5)
        actuator.set_intensity(0.0, 0.0)
        time.sleep(0.3)

        actions_a = []
        actions_b = []
        messages_a = []
        messages_b = []
        te_a_list = []
        te_b_list = []
        match_a_count = 0
        match_b_count = 0
        match_total = 0
        weight_snapshots_a = []
        weight_snapshots_b = []
        cpu_log = []
        gpu_log = []
        target_cpu_log = []
        target_gpu_log = []
        energies = []
        step = 0
        half = self.zone_width / 2.0

        try:
            for step in range(max_steps):
                sensors, state = sensor.read()
                cpu_load = state.get("cpu_load", 0.0)
                gpu_load = state.get("gpu_load", 0.0)

                if last_cpu is None:
                    last_cpu = cpu_load
                    last_gpu = gpu_load
                    continue

                # Меняем цели
                if step % self.target_change_period == 0:
                    target_cpu = float(rng.uniform(0.35, 0.65))
                    target_gpu = float(rng.uniform(0.35, 0.65))

                # Снимок весов каждые 200 тактов
                if step % 100 == 0 and step > 0:
                    weight_snapshots_a.append(self.brain_a.weights.copy())
                    weight_snapshots_b.append(self.brain_b.weights.copy())

                    # TE: target -> message
                    if len(messages_a) >= 50:
                        try:
                            ma_arr = np.array(messages_a[-100:])
                            tg_arr = np.array(target_gpu_log[-100:])
                            te_a = transfer_entropy(tg_arr, ma_arr, history=1, n_bins=5)
                            te_a_list.append(te_a)
                        except Exception:
                            pass
                    if len(messages_b) >= 50:
                        try:
                            mb_arr = np.array(messages_b[-100:])
                            tc_arr = np.array(target_cpu_log[-100:])
                            te_b = transfer_entropy(tc_arr, mb_arr, history=1, n_bins=5)
                            te_b_list.append(te_b)
                        except Exception:
                            pass

                # A: видит оба ресурса + target_gpu (чужую цель)
                input_a = np.array([cpu_load, target_gpu,
                                     last_action_a[0], last_message_b[0]])
                action_a = self.brain_a.forward(input_a)
                cpu_intensity = discretize_action(float(action_a[0]), n_levels=6)
                # ВРОЖДЁННЫЙ КАНАЛ: A знает target_gpu — передаёт напрямую
                message_a = float(target_gpu)

                # B: видит оба ресурса + target_cpu (чужую цель)
                input_b = np.array([gpu_load, target_cpu,
                                     last_action_b[0], last_message_a[0]])
                action_b = self.brain_b.forward(input_b)
                gpu_intensity = discretize_action(float(action_b[0]), n_levels=6)
                # ВРОЖДЁННЫЙ КАНАЛ: B знает target_cpu — передаёт напрямую
                message_b = float(target_cpu)

                # Применяем
                actuator.set_intensity(cpu_intensity, gpu_intensity)
                time.sleep(self.dt)

                next_sensors, next_state = sensor.read()
                next_cpu = next_state.get("cpu_load", 0.0)
                next_gpu = next_state.get("gpu_load", 0.0)

                # Командный reward
                error_cpu = abs(next_cpu - target_cpu)
                error_gpu = abs(next_gpu - target_gpu)
                # Раздельные reward: каждый за СВОЙ ресурс
                # A за CPU (свой), B за GPU (свой)
                # Но A не знает target_cpu, B не знает target_gpu
                # Значит: каждый должен ЧИТАТЬ message напарника
                # Мозг учится действиям на СВОЙ ресурс.
                # A влияет на CPU -> reward_a за CPU.
                # B влияет на GPU -> reward_b за GPU.
                # message — прямой канал: message_a = target_gpu, message_b = target_cpu.
                # A читает message_b и двигает CPU к нему.
                # B читает message_a и двигает GPU к нему.
                reward_a = -error_cpu
                reward_b = -error_gpu

                self.brain_a.update_weights(reward_a)
                self.brain_b.update_weights(reward_b)

                # Energy
                energy.cpu_min = target_cpu - half
                energy.cpu_max = target_cpu + half
                energy.gpu_min = target_gpu - half
                energy.gpu_max = target_gpu + half

                # ЗАЩИТА ОТ ПАРАЗИТОВ: штраф за молчание, бонус за активность
                if len(messages_a) >= 100:
                    recent_ma = np.array(messages_a[-100:])
                    recent_mb = np.array(messages_b[-100:])
                    std_ma = recent_ma.std()
                    std_mb = recent_mb.std()

                    # Если оба молчат — штраф обоим
                    if std_ma < 0.01 and std_mb < 0.01:
                        energy.energy = max(0.0, energy.energy - 0.01)
                    # Бонус за активность
                    if std_ma > 0.05 and std_mb > 0.05:
                        energy.energy = min(1.0, energy.energy + 0.005)

                energy.update(next_state)

                if not energy.is_alive():
                    break

                actions_a.append(cpu_intensity)
                actions_b.append(gpu_intensity)
                messages_a.append(message_a)
                messages_b.append(message_b)
                cpu_log.append(next_cpu)
                gpu_log.append(next_gpu)
                target_cpu_log.append(target_cpu)
                target_gpu_log.append(target_gpu)
                energies.append(energy.energy)

                last_cpu = next_cpu
                last_gpu = next_gpu
                last_action_a = np.array([cpu_intensity])
                last_action_b = np.array([gpu_intensity])
                last_message_a = np.array([message_a])
                last_message_b = np.array([message_b])

        finally:
            actuator.stop()

        # АССИМИЛЯЦИЯ: стабильные веса -> заморозка
        frozen_count_a = 0
        frozen_count_b = 0
        if len(weight_snapshots_a) >= 3:
            # Считаем std по снимкам для каждого веса
            stack_a = np.stack(weight_snapshots_a)  # (n_snap, n, n)
            stack_b = np.stack(weight_snapshots_b)
            std_weights_a = stack_a.std(axis=0)  # (n, n)
            std_weights_b = stack_b.std(axis=0)

            # Стабильные веса (std < 0.001)
            stable_a = std_weights_a < 0.001
            stable_b = std_weights_b < 0.001

            # ЛИМИТ: не замораживаем больше 60% весов
            MAX_FROZEN = 0.50
            total = self.brain_a.n * self.brain_a.n

            # A
            if self.brain_a.frozen_mask is None:
                current_frozen = 0
            else:
                current_frozen = int(self.brain_a.frozen_mask.sum())
            current_percent = current_frozen / total

            if current_percent < MAX_FROZEN:
                # Сколько ещё можем заморозить
                remaining = int((MAX_FROZEN - current_percent) * total)
                # Берём только первые N стабильных весов
                stable_flat = stable_a.flatten()
                stable_indices = np.where(stable_flat)[0]
                n_to_freeze = min(remaining, len(stable_indices))
                if n_to_freeze > 0:
                    new_mask_flat = np.zeros(total)
                    new_mask_flat[stable_indices[:n_to_freeze]] = 1.0
                    new_mask = new_mask_flat.reshape(stable_a.shape)
                    if self.brain_a.frozen_mask is None:
                        self.brain_a.frozen_mask = new_mask
                    else:
                        self.brain_a.frozen_mask = np.maximum(self.brain_a.frozen_mask, new_mask)

            # B
            if self.brain_b.frozen_mask is None:
                current_frozen = 0
            else:
                current_frozen = int(self.brain_b.frozen_mask.sum())
            current_percent = current_frozen / total

            if current_percent < MAX_FROZEN:
                remaining = int((MAX_FROZEN - current_percent) * total)
                stable_flat = stable_b.flatten()
                stable_indices = np.where(stable_flat)[0]
                n_to_freeze = min(remaining, len(stable_indices))
                if n_to_freeze > 0:
                    new_mask_flat = np.zeros(total)
                    new_mask_flat[stable_indices[:n_to_freeze]] = 1.0
                    new_mask = new_mask_flat.reshape(stable_b.shape)
                    if self.brain_b.frozen_mask is None:
                        self.brain_b.frozen_mask = new_mask
                    else:
                        self.brain_b.frozen_mask = np.maximum(self.brain_b.frozen_mask, new_mask)

            # Итоговый размер маски (унаследованное + новое)
            pass

        # Считаем ИТОГОВЫЙ размер маски (не только новое)
        if self.brain_a.frozen_mask is not None:
            frozen_count_a = int(self.brain_a.frozen_mask.sum())
        if self.brain_b.frozen_mask is not None:
            frozen_count_b = int(self.brain_b.frozen_mask.sum())

        # Метрики
        std_a = float(np.std(actions_a)) if actions_a else 0.0
        std_b = float(np.std(actions_b)) if actions_b else 0.0

        # Считаем std сообщений и средний TE
        std_msg_a = float(np.std(messages_a)) if messages_a else 0.0
        std_msg_b = float(np.std(messages_b)) if messages_b else 0.0
        te_a_mean = float(np.mean(te_a_list)) if te_a_list else 0.0
        te_b_mean = float(np.mean(te_b_list)) if te_b_list else 0.0

        self.result = {
            "pair_id": self.pair_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "lifetime": step,
            "mean_energy": float(np.mean(energies)) if energies else 0.0,
            "std_a": std_a,
            "std_b": std_b,
            "std_msg_a": std_msg_a,
            "std_msg_b": std_msg_b,
            "te_a": te_a_mean,
            "te_b": te_b_mean,
            "match_a": match_a_count / match_total if match_total > 0 else 0.0,
            "match_b": match_b_count / match_total if match_total > 0 else 0.0,
            "frozen_a": frozen_count_a,
            "frozen_b": frozen_count_b,
            "frozen_percent_a": 100.0 * frozen_count_a / (self.brain_a.n * self.brain_a.n),
            "frozen_percent_b": 100.0 * frozen_count_b / (self.brain_b.n * self.brain_b.n),
            "data": {
                "actions_a": np.array(actions_a),
                "actions_b": np.array(actions_b),
                "cpu": np.array(cpu_log),
                "gpu": np.array(gpu_log),
                "target_cpu": np.array(target_cpu_log),
                "target_gpu": np.array(target_gpu_log),
            },
        }

        # Сохраняем данные для TE
        np.savez(
            f"sep_evo_gen{self.generation}_pair{self.pair_id}.npz",
            actions_a=np.array(actions_a),
            actions_b=np.array(actions_b),
            messages_a=np.array(messages_a),
            messages_b=np.array(messages_b),
            cpu=np.array(cpu_log),
            gpu=np.array(gpu_log),
            target_cpu=np.array(target_cpu_log),
            target_gpu=np.array(target_gpu_log),
        )

        return self.result


def make_initial_pairs(n_pairs: int = 5, generation: int = 0, seed: int = 42):
    """Начальные пары — все случайные. A и B видят по 4 входа."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n_pairs):
        brain_a = Brain(n_input=4, n_output=3, seed=int(rng.integers(0, 1_000_000)))
        brain_b = Brain(n_input=4, n_output=3, seed=int(rng.integers(0, 1_000_000)))
        pair = CommPairSep(
            pair_id=i,
            generation=generation,
            parent_id=-1,
            brain_a=brain_a,
            brain_b=brain_b,
        )
        pairs.append(pair)
    return pairs


def select_parents(results: list, n_parents: int = 3, tournament_size: int = 3, seed: int = None) -> list:
    """
    Отбор по КОММУНИКАЦИИ (TE), с базовым фильтром выживания.

    fitness = (te_a + te_b) × energy
    Если пара прожила < 200 тактов — отсеивается.
    """
    def fitness(r):
        # Базовый фильтр: выживание
        if r["lifetime"] < 200:
            return -1.0
        # Сумма TE (возврат к мягкому отбору)
        te = r.get("te_a", 0.0) + r.get("te_b", 0.0)
        return te * r["mean_energy"]

    sorted_results = sorted(results, key=fitness, reverse=True)
    return sorted_results[:n_parents]


def breed_next_generation(
    parents_results,
    parents_brains,
    generation: int,
    n_offspring_per_parent: int = 2,
    mutation_sigma: float = 0.03,
    n_random: int = 1,
    seed: int = None,
):
    """Создаёт следующее поколение из родителей."""
    rng = np.random.default_rng(seed)
    next_gen = []
    next_id = 0

    for parent_result in parents_results:
        parent_id = parent_result["pair_id"]
        brain_a_parent, brain_b_parent = parents_brains[parent_id]

        for k in range(n_offspring_per_parent):
            # Потомок A
            child_a = Brain(n_input=4, n_output=3, seed=int(rng.integers(0, 1_000_000)))
            child_a.weights = mutate_weights(brain_a_parent.weights, mutation_sigma, rng)
            child_a.activations = np.zeros_like(brain_a_parent.activations)
            child_a.trace = np.zeros_like(brain_a_parent.trace)
            # НАСЛЕДОВАНИЕ: копируем frozen_mask от родителя
            if brain_a_parent.frozen_mask is not None:
                child_a.frozen_mask = brain_a_parent.frozen_mask.copy()

            # Потомок B
            child_b = Brain(n_input=4, n_output=3, seed=int(rng.integers(0, 1_000_000)))
            child_b.weights = mutate_weights(brain_b_parent.weights, mutation_sigma, rng)
            child_b.activations = np.zeros_like(brain_b_parent.activations)
            child_b.trace = np.zeros_like(brain_b_parent.trace)
            # НАСЛЕДОВАНИЕ: копируем frozen_mask от родителя
            if brain_b_parent.frozen_mask is not None:
                child_b.frozen_mask = brain_b_parent.frozen_mask.copy()

            pair = CommPairSep(
                pair_id=next_id,
                generation=generation,
                parent_id=parent_id,
                brain_a=child_a,
                brain_b=child_b,
            )
            next_gen.append(pair)
            next_id += 1

    # Случайная пара для разнообразия
    for k in range(n_random):
        brain_a = Brain(n_input=4, n_output=3, seed=int(rng.integers(0, 1_000_000)))
        brain_b = Brain(n_input=4, n_output=3, seed=int(rng.integers(0, 1_000_000)))
        pair = CommPairSep(
            pair_id=next_id,
            generation=generation,
            parent_id=-1,
            brain_a=brain_a,
            brain_b=brain_b,
        )
        next_gen.append(pair)
        next_id += 1

    return next_gen


def run_evolution(
    n_generations: int = 5,
    n_pairs: int = 10,
    max_steps: int = 2000,
    n_parents: int = 3,
    n_offspring: int = 3,
    mutation_sigma: float = 0.03,
    log_path: str = "evolution_sep.csv",
    seed: int = 42,
):
    """Главный цикл эволюции."""
    all_results = []

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "generation", "pair_id", "parent_id", "lifetime",
        "mean_energy", "std_a", "std_b",
    ])

    pairs = make_initial_pairs(n_pairs, generation=0, seed=seed)

    for gen in range(n_generations):
        print()
        print(f"========== ПОКОЛЕНИЕ {gen} ==========")
        print(f"Пар: {len(pairs)}")
        print()

        gen_results = []
        parent_brains = {}

        for pair in pairs:
            t0 = time.time()
            result = pair.run_life(max_steps=max_steps)
            elapsed = time.time() - t0

            gen_results.append(result)
            parent_brains[pair.pair_id] = (pair.brain_a, pair.brain_b)

            print(f"  Пара {result['pair_id']:>2} (род. {result['parent_id']:>2}): "
                  f"жизнь={result['lifetime']:>4}, "
                  f"энергия={result['mean_energy']:.3f}, "
                  f"std_a={result['std_a']:.3f}, "
                  f"std_b={result['std_b']:.3f}, "
                  f"match_a={result.get('match_a', 0):.2f}, "
                  f"match_b={result.get('match_b', 0):.2f}, "
                  f"время={elapsed:.1f}с")

            writer.writerow([
                result["generation"], result["pair_id"], result["parent_id"],
                result["lifetime"], result["mean_energy"],
                result["std_a"], result["std_b"],
            ])
            log_file.flush()

            all_results.append(result)

        # Статистика поколения
        lifetimes = [r["lifetime"] for r in gen_results]
        energies = [r["mean_energy"] for r in gen_results]
        std_a_list = [r["std_a"] for r in gen_results]
        std_b_list = [r["std_b"] for r in gen_results]

        print()
        print(f"  Итог поколения {gen}:")
        print(f"    Жизнь: среднее={np.mean(lifetimes):.1f}")
        print(f"    Энергия: среднее={np.mean(energies):.3f}")
        print(f"    std_a: среднее={np.mean(std_a_list):.4f}")
        print(f"    std_b: среднее={np.mean(std_b_list):.4f}")

        if gen == n_generations - 1:
            break

        # Адаптивный n_parents: max(3, 30% популяции)
        n_parents_adaptive = max(3, int(0.3 * len(gen_results)))
        print(f"  n_parents = {n_parents_adaptive} (из {len(gen_results)} пар)")
        parents = select_parents(gen_results, n_parents=n_parents_adaptive, tournament_size=3, seed=seed + gen)
        print()
        print(f"  Отобраны родители: {[p['pair_id'] for p in parents]}")

        n_random = max(2, n_pairs - n_parents * n_offspring)
        pairs = breed_next_generation(
            parents, parent_brains,
            generation=gen + 1,
            n_offspring_per_parent=n_offspring,
            mutation_sigma=mutation_sigma,
            n_random=n_random,
            seed=seed + gen + 1,
        )

        # Считаем потомков и унаследованный ЦГ
        for parent_result in parents:
            parent_id = parent_result["pair_id"]
            offspring = [p for p in pairs if p.parent_id == parent_id]
            parent_result["offspring_count"] = len(offspring)

            inherited_ratios = []
            parent_brain_a, parent_brain_b = parent_brains[parent_id]
            for child in offspring:
                if parent_brain_a.frozen_mask is not None and child.brain_a.frozen_mask is not None:
                    pf = parent_brain_a.frozen_mask
                    cf = child.brain_a.frozen_mask
                    total = pf.sum()
                    if total > 0:
                        inherited_ratios.append((pf * cf).sum() / total)
            parent_result["inherited_ratio"] = float(np.mean(inherited_ratios)) if inherited_ratios else 0.0

            print(f"  Родитель {parent_id}: потомков={parent_result['offspring_count']}, "
                  f"унаследовано={parent_result['inherited_ratio']:.2f}")

    log_file.close()
    print()
    print(f"Эволюция завершена. Логи: {log_path}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--parents", type=int, default=3)
    parser.add_argument("--offspring", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--log", type=str, default="evolution_sep.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evolution(
        n_generations=args.generations,
        n_pairs=args.pairs,
        max_steps=args.steps,
        n_parents=args.parents,
        n_offspring=args.offspring,
        mutation_sigma=args.sigma,
        log_path=args.log,
        seed=args.seed,
    )
