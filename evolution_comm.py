"""
Фаза 4: эволюция коммуникации.

Пары (A, B) — "организмы". Отбор по совместному выживанию.
Наследование весов обоих. Мутации у обоих.

Вопрос: усилится ли связь A ↔ B с поколениями?
"""

import sys
import csv
import time
import numpy as np

sys.path.insert(0, '.')

from brain import Brain
from world_model import WorldModel
from body_sensor import BodySensor
from body_actuator import BodyActuator
from energy import Energy
from mutation import mutate_weights


class CommPair:
    """Пара агентов (A, B) — один организм."""

    def __init__(
        self,
        pair_id: int,
        generation: int,
        parent_id: int,
        brain_a: Brain,
        brain_b: Brain,
        dt: float = 0.05,
    ):
        self.pair_id = pair_id
        self.generation = generation
        self.parent_id = parent_id
        self.brain_a = brain_a
        self.brain_b = brain_b
        self.dt = dt
        self.result = None

    def run_life(self, max_steps: int = 3000):
        """Пара живёт до смерти или до лимита."""
        sensor = BodySensor()
        actuator = BodyActuator()
        energy = Energy(
            drain_rate=0.003,
            bounty_rate=0.003,
            bounty_probability=0.7,
            cpu_min=0.3, cpu_max=0.7,
            gpu_min=0.3, gpu_max=0.7,
            death_threshold=0.2,
            drift_period=500,
        )

        model_a = WorldModel(seed=42)
        model_b = WorldModel(seed=43)

        last_action_a = np.array([0.0, 0.0])
        last_action_b = np.array([0.0, 0.0])

        actuator.start()
        # Прогрев
        actuator.set_intensity(0.3, 0.3)
        time.sleep(1.5)
        actuator.set_intensity(0.0, 0.0)
        time.sleep(0.3)

        errors_a = []
        errors_b = []
        energies = []
        actions_a = []
        actions_b = []
        step = 0

        try:
            last_sensors_a = None
            last_sensors_b = None

            for step in range(max_steps):
                sensors, state = sensor.read()

                if last_sensors_a is None:
                    last_sensors_a = sensors
                    last_sensors_b = sensors
                    continue

                # A действует
                input_a = np.concatenate([sensors, last_action_a])
                action_a = self.brain_a.forward(input_a)

                actuator.set_intensity(action_a[0], action_a[1])
                time.sleep(self.dt / 2.0)

                sensors_mid, state_mid = sensor.read()

                model_a.predict(last_sensors_a, action_a)
                error_a = model_a.update(sensors_mid)
                self.brain_a.update_weights(-error_a)

                # B действует
                input_b = np.concatenate([sensors_mid, last_action_b])
                action_b = self.brain_b.forward(input_b)

                actuator.set_intensity(action_b[0], action_b[1])
                time.sleep(self.dt / 2.0)

                sensors_end, state_end = sensor.read()

                model_b.predict(last_sensors_b, action_b)
                error_b = model_b.update(sensors_end)
                self.brain_b.update_weights(-error_b)

                # Каждые 100 тактов пересчитываем связь A↔B
                if len(actions_a) >= 100 and len(actions_a) % 100 == 0:
                    arr_a = np.array(actions_a[-100:])
                    arr_b = np.array(actions_b[-100:])
                    corr = np.corrcoef(arr_a[:, 0], arr_b[:, 0])[0, 1]
                    if not np.isnan(corr):
                        energy.connection_strength = float(np.clip(abs(corr), 0.0, 1.0))

                # Энергия
                energy.update(state_end)
                if not energy.is_alive():
                    break

                # Запоминаем
                errors_a.append(error_a)
                errors_b.append(error_b)
                energies.append(energy.energy)
                actions_a.append(action_a.copy())
                actions_b.append(action_b.copy())

                last_sensors_a = sensors_end
                last_sensors_b = sensors_mid
                last_action_a = action_a
                last_action_b = action_b

        finally:
            actuator.stop()

        # Метрики
        if len(actions_a) > 50:
            arr_a = np.array(actions_a)
            arr_b = np.array(actions_b)
            corr_a_to_b = np.corrcoef(arr_a[:-1, 0], arr_b[1:, 0])[0, 1]
            corr_b_to_a = np.corrcoef(arr_b[:-1, 0], arr_a[1:, 0])[0, 1]
        else:
            corr_a_to_b = 0.0
            corr_b_to_a = 0.0

        self.result = {
            "pair_id": self.pair_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "lifetime": step,
            "mean_energy": float(np.mean(energies)) if energies else 0.0,
            "final_energy": float(energies[-1]) if energies else 0.0,
            "mean_error_a": float(np.mean(errors_a)) if errors_a else 0.0,
            "mean_error_b": float(np.mean(errors_b)) if errors_b else 0.0,
            "corr_a_to_b": float(corr_a_to_b),
            "corr_b_to_a": float(corr_b_to_a),
        }
        return self.result


def make_initial_pairs(n_pairs: int = 5, generation: int = 0, seed: int = 42):
    """Создаёт начальные пары — все случайные."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n_pairs):
        brain_a = Brain(seed=int(rng.integers(0, 1_000_000)))
        brain_b = Brain(seed=int(rng.integers(0, 1_000_000)))
        pair = CommPair(
            pair_id=i,
            generation=generation,
            parent_id=-1,
            brain_a=brain_a,
            brain_b=brain_b,
        )
        pairs.append(pair)
    return pairs


def select_parents(results: list, n_parents: int = 2) -> list:
    """Отбор по fitness = lifetime * mean_energy."""
    def fitness(r):
        corr = abs(r["corr_a_to_b"]) + abs(r["corr_b_to_a"])
        return r["lifetime"] * r["mean_energy"] * (1.0 + 2.0 * corr)
    sorted_results = sorted(results, key=fitness, reverse=True)
    return sorted_results[:n_parents]


def breed_next_generation(
    parents_results,
    parents_brains,
    generation: int,
    n_offspring_per_parent: int = 2,
    mutation_sigma: float = 0.02,
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
            child_a = Brain(seed=int(rng.integers(0, 1_000_000)))
            child_a.weights = mutate_weights(brain_a_parent.weights, mutation_sigma, rng)
            child_a.activations = np.zeros_like(brain_a_parent.activations)
            child_a.trace = np.zeros_like(brain_a_parent.trace)
            child_a.reward_baseline = brain_a_parent.reward_baseline

            # Потомок B
            child_b = Brain(seed=int(rng.integers(0, 1_000_000)))
            child_b.weights = mutate_weights(brain_b_parent.weights, mutation_sigma, rng)
            child_b.activations = np.zeros_like(brain_b_parent.activations)
            child_b.trace = np.zeros_like(brain_b_parent.trace)
            child_b.reward_baseline = brain_b_parent.reward_baseline

            pair = CommPair(
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
        brain_a = Brain(seed=int(rng.integers(0, 1_000_000)))
        brain_b = Brain(seed=int(rng.integers(0, 1_000_000)))
        pair = CommPair(
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
    n_pairs: int = 5,
    max_steps: int = 3000,
    n_parents: int = 2,
    n_offspring: int = 2,
    mutation_sigma: float = 0.02,
    log_path: str = "evolution_comm.csv",
    seed: int = 42,
):
    """Главный цикл эволюции коммуникации."""
    all_results = []

    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "generation", "pair_id", "parent_id", "lifetime",
        "mean_energy", "final_energy", "mean_error_a", "mean_error_b",
        "corr_a_to_b", "corr_b_to_a",
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
                  f"A->B={result['corr_a_to_b']:+.3f}, "
                  f"B->A={result['corr_b_to_a']:+.3f}, "
                  f"время={elapsed:.1f}с")

            writer.writerow([
                result["generation"], result["pair_id"], result["parent_id"],
                result["lifetime"], result["mean_energy"], result["final_energy"],
                result["mean_error_a"], result["mean_error_b"],
                result["corr_a_to_b"], result["corr_b_to_a"],
            ])
            log_file.flush()

            all_results.append(result)

        # Статистика поколения
        lifetimes = [r["lifetime"] for r in gen_results]
        energies = [r["mean_energy"] for r in gen_results]
        corr_ab = [r["corr_a_to_b"] for r in gen_results]
        corr_ba = [r["corr_b_to_a"] for r in gen_results]

        print()
        print(f"  Итог поколения {gen}:")
        print(f"    Жизнь: среднее={np.mean(lifetimes):.1f}")
        print(f"    Энергия: среднее={np.mean(energies):.3f}")
        print(f"    A->B: среднее={np.mean(corr_ab):+.3f}")
        print(f"    B->A: среднее={np.mean(corr_ba):+.3f}")

        if gen == n_generations - 1:
            break

        parents = select_parents(gen_results, n_parents=n_parents)
        print()
        print(f"  Отобраны родители: {[p['pair_id'] for p in parents]}")

        n_random = max(1, n_pairs - n_parents * n_offspring)
        pairs = breed_next_generation(
            parents, parent_brains,
            generation=gen + 1,
            n_offspring_per_parent=n_offspring,
            mutation_sigma=mutation_sigma,
            n_random=n_random,
            seed=seed + gen + 1,
        )

    log_file.close()
    print()
    print(f"Эволюция завершена. Логи: {log_path}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--parents", type=int, default=2)
    parser.add_argument("--offspring", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--log", type=str, default="evolution_comm.csv")
    args = parser.parse_args()

    run_evolution(
        n_generations=args.generations,
        n_pairs=args.pairs,
        max_steps=args.steps,
        n_parents=args.parents,
        n_offspring=args.offspring,
        mutation_sigma=args.sigma,
        log_path=args.log,
    )
