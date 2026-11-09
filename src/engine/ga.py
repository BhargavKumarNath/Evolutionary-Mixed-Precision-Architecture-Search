import json
import logging
import os
import random
from dataclasses import asdict
from typing import Any, cast

import wandb

from ..core.configuration import configuration_hash
from ..core.evaluator import Evaluator, FitnessMetrics
from ..core.search_space import Genome, MixedPrecisionSearchSpace, SearchSpace
from ..utils.reproducibility import capture_rng_state, restore_rng_state
from .operators import crossover_uniform, mutate_random_gene
from .pareto import crowded_comparison_key, crowding_distance, fast_nondominated_sort

logger = logging.getLogger(__name__)

class GAEngine:
    def __init__(self, 
                 search_space: SearchSpace, 
                 evaluator: Evaluator,
                 config: dict[str, Any],
                 tracking_config: dict[str, Any] | None = None,
                 output_dir: str = "./data/logs"):
        
        self.space = search_space
        self.evaluator = evaluator
        self.cfg = config
        self.tracking_cfg = tracking_config or {"enabled": False}
        self.output_dir = output_dir
        
        self.population: list[Genome] = []
        self.fitness_cache: dict[str, FitnessMetrics] = {}
        self.history: list[dict] = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize WandB if enabled
        if self.tracking_cfg.get("enabled", False):
            wandb.init(
                project=self.tracking_cfg.get("project", "empas"),
                config={**self.cfg, "search_space": self.space.__class__.__name__},
                tags=self.tracking_cfg.get("tags", []),
                dir=self.output_dir
            )

    def _get_fitness_str(self, genome: Genome) -> str:
        return str(genome.genes)

    def evaluate_population(self, population: list[Genome]) -> list[FitnessMetrics]:
        fitnesses = []
        for genome in population:
            if not self.space.validate(genome):
                raise ValueError(f"Invalid genome for configured search space: {genome.genes}")
            key = self._get_fitness_str(genome)
            if key in self.fitness_cache:
                fitnesses.append(self.fitness_cache[key])
            else:
                fit = self.evaluator.evaluate(genome)
                self.fitness_cache[key] = fit
                fitnesses.append(fit)
        return fitnesses

    def _feasible_or(self, child: Genome, parent: Genome) -> Genome:
        """Repair an infeasible child into the average-bit range; fall back to the parent."""
        if self.space.validate(child):
            return child
        repair = getattr(self.space, "repair", None)
        if repair is not None:
            try:
                fixed = repair(child)
                if self.space.validate(fixed):
                    return fixed
            except RuntimeError:
                pass
        return parent

    def select_tournament(
        self,
        population: list[Genome],
        rank: dict[int, int],
        distance: dict[int, float],
    ) -> Genome:
        """NSGA-II crowded binary/k-way tournament: lower front rank wins;
        ties within the same front are broken by larger crowding distance
        (preferring the less-crowded, more diverse individual)."""
        indices = random.sample(range(len(population)), self.cfg['tournament_size'])
        best_idx = min(indices, key=lambda idx: crowded_comparison_key(idx, rank, distance))
        return population[best_idx]

    def save_checkpoint(self, generation: int, stats: dict[str, Any]) -> None:
        """Save population, fitness, RNG, and experiment identity for resume."""
        filename = os.path.join(self.output_dir, f"checkpoint_gen_{generation}.json")

        population_data = [g.genes for g in self.population]
        data = {
            "schema_version": 1,
            "generation": generation,
            "stats": stats,
            "population": population_data,
            "fitness_cache": {
                key: asdict(value) for key, value in self.fitness_cache.items()
            },
            "history": self.history,
            "rng_state": capture_rng_state(),
            "config_hash": configuration_hash(self.cfg),
            "profile_identity": self._profile_identity(),
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load_checkpoint(self, path: str) -> int:
        """Restore a checkpoint and return the next generation to execute."""
        with open(path, "r") as handle:
            data = json.load(handle)
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema version")
        expected_hash = configuration_hash(self.cfg)
        if data.get("config_hash") != expected_hash:
            raise ValueError("Checkpoint configuration does not match current configuration")

        self.population = [Genome(genes=genes) for genes in data["population"]]
        if not all(self.space.validate(genome) for genome in self.population):
            raise ValueError("Checkpoint contains a genome invalid for current search space")
        self.fitness_cache = {
            key: FitnessMetrics(**value)
            for key, value in data.get("fitness_cache", {}).items()
        }
        self.history = data.get("history", [])
        restore_rng_state(data["rng_state"])
        return int(data["generation"]) + 1

    def _profile_identity(self) -> str | None:
        profile_path = self.cfg.get("profile_path")
        if profile_path is None:
            return None
        return str(profile_path)

    def step(self, generation: int):
        """One NSGA-II generation: rank parents, breed offspring via crowded
        tournament selection, then select the next generation from the
        combined parent+offspring pool by front rank and crowding distance
        (Deb et al., 2002)."""
        logger.info(f"--- Generation {generation} ---")
        pop_size = self.cfg['pop_size']

        # 1. Evaluate and rank the current (parent) population. Rank/distance
        #    drive crowded tournament selection when breeding offspring.
        parent_fitnesses = self.evaluate_population(self.population)
        parent_fronts = fast_nondominated_sort(parent_fitnesses)
        parent_rank = {idx: front_no for front_no, front in enumerate(parent_fronts) for idx in front}
        parent_distance: dict[int, float] = {}
        for front in parent_fronts:
            parent_distance.update(crowding_distance(front, parent_fitnesses))

        # 2. Breed offspring: crowded tournament selection + crossover + mutation
        offspring: list[Genome] = []
        mutation_space = cast("MixedPrecisionSearchSpace", self.space)
        while len(offspring) < pop_size:
            p1 = self.select_tournament(self.population, parent_rank, parent_distance)
            p2 = self.select_tournament(self.population, parent_rank, parent_distance)

            if random.random() < self.cfg['crossover_rate']:
                c1, c2 = crossover_uniform(p1, p2)
            else:
                c1, c2 = p1, p2

            c1 = mutate_random_gene(c1, mutation_space, self.cfg['mutation_rate'])
            c2 = mutate_random_gene(c2, mutation_space, self.cfg['mutation_rate'])

            c1 = self._feasible_or(c1, p1)
            c2 = self._feasible_or(c2, p2)

            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        # 3. mu+lambda: combine parents and offspring, then rank the union
        offspring_fitnesses = self.evaluate_population(offspring)
        combined_population = self.population + offspring
        combined_fitnesses = parent_fitnesses + offspring_fitnesses
        combined_fronts = fast_nondominated_sort(combined_fitnesses)

        # 4. Environmental selection: fill the next generation front-by-front;
        #    the last, partially-included front is trimmed by crowding
        #    distance so the population stays spread across the trade-off
        #    surface instead of clustering.
        next_indices: list[int] = []
        for front in combined_fronts:
            if len(next_indices) + len(front) <= pop_size:
                next_indices.extend(front)
            else:
                remaining = pop_size - len(next_indices)
                front_distance = crowding_distance(front, combined_fitnesses)
                ranked_front = sorted(front, key=lambda idx: -front_distance[idx])
                next_indices.extend(ranked_front[:remaining])
                break

        self.population = [combined_population[i] for i in next_indices]

        # 5. Stats & logging. Front 0 of the parent+offspring union is this
        #    generation's actual Pareto front (num_elites is its size).
        pareto_front0 = combined_fronts[0]
        losses = [f.validation_loss for f in combined_fitnesses]
        vrams = [f.vram_peak_mb for f in combined_fitnesses]

        stats = {
            "gen": generation,
            "min_loss": min(losses),
            "avg_loss": sum(losses) / len(losses),
            "min_vram": min(vrams),
            "avg_vram": sum(vrams) / len(vrams),
            "num_elites": len(pareto_front0),
        }

        logger.info(f"Stats: Loss={stats['min_loss']:.4f}, VRAM={stats['min_vram']:.0f}MB, Elites={stats['num_elites']}")
        self.history.append(stats)

        # WandB Logging
        if self.tracking_cfg.get("enabled", False):
            wandb.log(stats, step=generation)

            pareto_set = set(pareto_front0)
            table_data = []
            for i, (genome, fit) in enumerate(zip(combined_population, combined_fitnesses)):
                is_pareto = i in pareto_set
                table_data.append([fit.validation_loss, fit.vram_peak_mb, fit.latency_ms, str(genome.genes), is_pareto])

            table = wandb.Table(data=table_data, columns=["loss", "vram", "latency", "genes", "is_pareto"])

            wandb.log({
                "population_plot": wandb.plot.scatter(
                    table, "vram", "loss", title=f"Gen {generation}: Loss vs VRAM"
                )
            }, step=generation)

        checkpoint_freq = int(self.cfg.get("checkpoint_freq", 1))
        if generation % checkpoint_freq == 0:
            self.save_checkpoint(generation, stats)

    def run(self, resume_from: str | None = None) -> None:
        if resume_from is None:
            logger.info("Initializing population...")
            self.population = [self.space.sample() for _ in range(self.cfg["pop_size"])]
            start_generation = 1
        else:
            start_generation = self.load_checkpoint(resume_from)
            logger.info("Resuming search at generation %s", start_generation)

        for gen in range(start_generation, self.cfg["n_generations"] + 1):
            self.step(gen)
            
        if self.tracking_cfg.get("enabled", False):
            wandb.finish()
        logger.info("Search Complete.")