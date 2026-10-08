"""
Эволюция агентов: поколения, отбор, размножение.

Логика:
1. Поколение N: 10 агентов, каждый живёт до смерти или max_steps.
2. Записываем результаты: время жизни, средняя энергия, средняя ошибка.
3. Отбор: топ-3 по времени жизни.
4. Размножение: каждый родитель даёт 3 потомка с мутацией.
5. Поколение N+1: 9 потомков + 1 случайный агент.
6. Повтор.
"""

import sys
import os
import csv
import time
import numpy as np

sys.path.insert(0, '.')

from agent import Agent
from brain import Brain
from mutation import mutate_weights, crossover


class EvolvedAgent:
    def __init__(self, agent_id, generation, parent_id, brain):
        self.agent_id = agent_id
        self.generation = generation
        self.parent_id = parent_id
        self.brain = brain  # готовый мозг
        self.result = None

    def run_life(self, max_steps=1000, dt=0.05):
        """Один агент живёт свою жизнь. Возвращает статистику."""
        agent = Agent(dt=dt)
        # Заменяем мозг на свой
        agent.brain = self.brain

        agent.start()
        try:
            for i in range(max_steps):
                log = agent.step()
                if not log["alive"]:
                    break
        finally:
            agent.stop()

        errors = np.array(agent.errors) if agent.errors else np.array([0.0])
        energies = np.array(agent.energies) if agent.energies else np.array([0.0])

        self.result = {
            "agent_id": self.agent_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "lifetime": agent.step_count,
            "mean_energy": float(energies.mean()),
            "final_energy": float(energies[-1]),
            "mean_error": float(errors.mean()),
            "last_error": float(errors[-100:].mean()) if len(errors) >= 100 else float(errors.mean()),
        }

        # Возвращаем финальный мозг, чтобы сохранить его
        self.brain = agent.brain
        return self.result


def select_parents(results: list, n_parents: int = 3) -> list:
    """
    Отбирает лучших по fitness = lifetime × mean_energy.
    Проживи долго И накопи энергию.
    """
    def fitness(r):
        return r["lifetime"] * r["mean_energy"]

    sorted_results = sorted(results, key=fitness, reverse=True)
    return sorted_results[:n_parents]


def breed_next_generation(parents, parent_brains, generation, n_offspring_per_parent=3,
                          mutation_sigma=0.02, n_random=1, seed=None):
    """
    Создаёт следующее поколение.
    - parents: список результатов родителей (словари)
    - parent_brains: словарь {agent_id: Brain} — мозги родителей
    - generation: номер нового поколения
    - mutation_sigma: сила мутации
    Возвращает список EvolvedAgent.
    """
    rng = np.random.default_rng(seed)
    next_gen = []
    next_id = 0

    for parent_result in parents:
        parent_id = parent_result["agent_id"]
        parent_brain = parent_brains[parent_id]

        for k in range(n_offspring_per_parent):
            # Создаём ребёнка
            child_brain = Brain(seed=int(rng.integers(0, 1_000_000)))

            # Кроссовер: смешиваем веса двух родителей
            other_parent_result = parents[rng.integers(0, len(parents))]
            other_parent_brain = parent_brains[other_parent_result["agent_id"]]

            child_weights = crossover(parent_brain.weights, other_parent_brain.weights, rng)

            # Мутация
            child_brain.weights = mutate_weights(child_weights, mutation_sigma, rng)
            child_brain.activations = np.zeros_like(parent_brain.activations)
            child_brain.trace = np.zeros_like(parent_brain.trace)
            child_brain.reward_baseline = parent_brain.reward_baseline

            agent = EvolvedAgent(
                agent_id=next_id,
                generation=generation,
                parent_id=parent_id,
                brain=child_brain,
            )
            next_gen.append(agent)
            next_id += 1

    # Добавляем случайных новичков (для разнообразия)
    for k in range(n_random):
        fresh_brain = Brain(seed=int(rng.integers(0, 1_000_000)))
        agent = EvolvedAgent(
            agent_id=next_id,
            generation=generation,
            parent_id=-1,  # случайный
            brain=fresh_brain,
        )
        next_gen.append(agent)
        next_id += 1

    return next_gen


def make_initial_generation(n_agents: int = 10, generation: int = 0, seed: int = 42):
    """
    Создаёт первое поколение — все случайные, без родителей.
    """
    rng = np.random.default_rng(seed)
    agents = []
    for i in range(n_agents):
        brain = Brain(seed=int(rng.integers(0, 1_000_000)))
        agent = EvolvedAgent(
            agent_id=i,
            generation=generation,
            parent_id=-1,
            brain=brain,
        )
        agents.append(agent)
    return agents


def run_evolution(
    n_generations: int = 3,
    n_agents: int = 10,
    max_steps: int = 1000,
    dt: float = 0.05,
    n_parents: int = 3,
    n_offspring: int = 3,
    mutation_sigma: float = 0.02,
    log_path: str = "evolution_log.csv",
    seed: int = 42,
):
    """
    Главный цикл эволюции.
    """
    all_results = []

    # Открываем CSV для логов
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "generation", "agent_id", "parent_id",
        "lifetime", "mean_energy", "final_energy",
        "mean_error", "last_error",
    ])

    # Создаём начальное поколение
    agents = make_initial_generation(n_agents, generation=0, seed=seed)

    for gen in range(n_generations):
        print()
        print(f"========== ПОКОЛЕНИЕ {gen} ==========")
        print(f"Агентов: {len(agents)}")
        print()

        gen_results = []
        parent_brains = {}

        for agent in agents:
            t0 = time.time()
            result = agent.run_life(max_steps=max_steps, dt=dt)
            elapsed = time.time() - t0

            gen_results.append(result)
            # Сохраняем финальный мозг (для размножения)
            parent_brains[agent.agent_id] = agent.brain

            print(f"  Агент {result['agent_id']:>2} (род. {result['parent_id']:>2}): "
                  f"жизнь={result['lifetime']:>4}, "
                  f"энергия={result['mean_energy']:.3f}, "
                  f"ошибка={result['mean_error']:.4f}, "
                  f"время={elapsed:.1f}с")

            writer.writerow([
                result["generation"], result["agent_id"], result["parent_id"],
                result["lifetime"], result["mean_energy"], result["final_energy"],
                result["mean_error"], result["last_error"],
            ])
            log_file.flush()

            all_results.append(result)

        # Статистика поколения
        lifetimes = [r["lifetime"] for r in gen_results]
        energies = [r["mean_energy"] for r in gen_results]
        errors = [r["mean_error"] for r in gen_results]

        print()
        print(f"  Итог поколения {gen}:")
        print(f"    Жизнь:   среднее={np.mean(lifetimes):.1f}, "
              f"макс={np.max(lifetimes)}, мин={np.min(lifetimes)}")
        print(f"    Энергия: среднее={np.mean(energies):.3f}")
        print(f"    Ошибка:  среднее={np.mean(errors):.4f}")

        # Последнее поколение — не размножаем
        if gen == n_generations - 1:
            break

        # Отбор
        parents = select_parents(gen_results, n_parents=n_parents)
        print()
        print(f"  Отобраны родители: "
              f"{[p['agent_id'] for p in parents]} "
              f"(жизнь: {[p['lifetime'] for p in parents]})")

        # Размножение
        n_random = max(1, n_agents - n_parents * n_offspring)
        agents = breed_next_generation(
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
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--parents", type=int, default=3)
    parser.add_argument("--offspring", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--log", type=str, default="evolution_log.csv")
    args = parser.parse_args()

    run_evolution(
        n_generations=args.generations,
        n_agents=args.agents,
        max_steps=args.steps,
        dt=args.dt,
        n_parents=args.parents,
        n_offspring=args.offspring,
        mutation_sigma=args.sigma,
        log_path=args.log,
    )
