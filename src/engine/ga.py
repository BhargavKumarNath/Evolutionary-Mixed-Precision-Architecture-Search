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
from .pareto import get_pareto_front

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

    def select_tournament(self, population: list[Genome], fitnesses: list[FitnessMetrics]) -> Genome:
        indices = random.sample(range(len(population)), self.cfg['tournament_size'])
        best_idx = indices[0]
        for idx in indices[1:]:
            if self._dominates(fitnesses[idx], fitnesses[best_idx]) or (
                not self._dominates(fitnesses[best_idx], fitnesses[idx])
                and random.random() < 0.5
            ):
                best_idx = idx
        return population[best_idx]

    def _dominates(self, f1: FitnessMetrics, f2: FitnessMetrics) -> bool:
        from .pareto import dominates
        return dominates(f1, f2)

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
        logger.info(f"--- Generation {generation} ---")
        
        # 1. Evaluate
        fitnesses = self.evaluate_population(self.population)
        
        # 2. Pareto Front
        pareto_indices = get_pareto_front(fitnesses)
        elites = [self.population[i] for i in pareto_indices]
        
        # 3. Stats & Logging
        losses = [f.validation_loss for f in fitnesses]
        vrams = [f.vram_peak_mb for f in fitnesses]
        
        stats = {
            "gen": generation,
            "min_loss": min(losses),
            "avg_loss": sum(losses) / len(losses),
            "min_vram": min(vrams),
            "avg_vram": sum(vrams) / len(vrams),
            "num_elites": len(elites)
        }
        
        logger.info(f"Stats: Loss={stats['min_loss']:.4f}, VRAM={stats['min_vram']:.0f}MB, Elites={stats['num_elites']}")
        self.history.append(stats)
        
        # WandB Logging
        if self.tracking_cfg.get("enabled", False):
            # Log metrics
            wandb.log(stats, step=generation)
            
            # Create a Scatter Plot (Loss vs VRAM)
            table_data = []
            for genome, fit in zip(self.population, fitnesses):
                is_pareto = genome in elites
                table_data.append([fit.validation_loss, fit.vram_peak_mb, fit.latency_ms, str(genome.genes), is_pareto])
            
            table = wandb.Table(data=table_data, columns=["loss", "vram", "latency", "genes", "is_pareto"])
            
            wandb.log({
                "population_plot": wandb.plot.scatter(
                    table, "vram", "loss", title=f"Gen {generation}: Loss vs VRAM"
                )
            }, step=generation)

        # 4. Evolution (Selection, Crossover, Mutation)
        next_pop = []
        
        # Elitism
        num_elites_to_keep = min(len(elites), self.cfg['pop_size'] // 2)
        random.shuffle(elites)
        next_pop.extend(elites[:num_elites_to_keep])
        
        while len(next_pop) < self.cfg['pop_size']:
            p1 = self.select_tournament(self.population, fitnesses)
            p2 = self.select_tournament(self.population, fitnesses)
            
            if random.random() < self.cfg['crossover_rate']:
                c1, c2 = crossover_uniform(p1, p2)
            else:
                c1, c2 = p1, p2
                
            mutation_space = cast("MixedPrecisionSearchSpace", self.space)
            c1 = mutate_random_gene(c1, mutation_space, self.cfg['mutation_rate'])
            c2 = mutate_random_gene(c2, mutation_space, self.cfg['mutation_rate'])

            if not self.space.validate(c1):
                c1 = p1
            if not self.space.validate(c2):
                c2 = p2
            
            next_pop.append(c1)
            if len(next_pop) < self.cfg['pop_size']:
                next_pop.append(c2)
        
        self.population = next_pop

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