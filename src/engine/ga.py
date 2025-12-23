import json
import os
import random
import time
from typing import List, Dict, Any
import logging
from dataclasses import asdict
from ..core.search_space import SearchSpace, Genome
from ..core.evaluator import Evaluator, FitnessMetrics
from .pareto import dominates, get_pareto_front
from .operators import crossover_uniform, mutate_random_gene
logger = logging.getLogger(__name__)

class GAEngine:
    def __init__(self, search_space: SearchSpace, evaluator: Evaluator, config: Dict[str, Any], output_dir: str = "./data/logs"):
        self.space = search_space
        self.evaluator = evaluator
        self.cfg = config
        self.output_dir = output_dir
        self.population: List[Genome] = []
        self.fitness_cache: Dict[str, FitnessMetrics] = {}
        self.history: List[Dict] = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_fitness_str(self, genome: Genome) -> str:
        """Create a unique string key for the genome for caching"""
        return str(genome.genes)
    
    def evaluate_population(self, population: List[Genome]) -> List[FitnessMetrics]:
        """Sequentially evaluate the population."""
        fitnesses = []
        for i, genome in enumerate(population):
            key = self._get_fitness_str(genome)

            if key in self.fitness_cache:
                fitnesses.append(self.fitness_cache[key])
            else:
                if i % 5 == 0:
                    logger.info(f"Evaluating candidate {i+1}/{len(population)}...")
                fit = self.evaluator.evaluate(genome)
                self.fitness_cache[key] = fit
                fitnesses.append(fit)
        
        return fitnesses
    
    def select_tournament(self, population: List[Genome], fitnesses: List[FitnessMetrics]) -> Genome:
        """Select one parent using Tournamenet Selection.
        Pick k individuals, return the one that dominates the others. If non-dominated, pick randomly among them"""
        indices = random.sample(range(len(population)), self.cfg['tournament_size'])
        best_idx = indices[0]

        # If challanger dominates current best, swap
        for idx in indices[1:]:
            if self._dominates(fitnesses[idx], fitnesses[best_idx]):
                best_idx = idx
            
            elif not self._dominates(fitnesses[best_idx], fitnesses[idx]):
                if random.random() < 0.5:
                    best_idx = idx
        
        return population[best_idx]
    
    def _dominates(self, f1: FitnessMetrics, f2: FitnessMetrics) -> bool:
        """Wrapper for dominance logic defined in pareto.py or inline"""
        return dominates(f1, f2)
    
    def step(self, generation: int):
        """Run one generation of evolution"""
        logger.info(f"--- Generation {generation} ---")

        # 1. Evaluate current population
        fitnesses = self.evaluate_population(self.population)

        # 2. Identify Elites (Pareto Front)
        pareto_indices = get_pareto_front(fitnesses)
        elites = [self.population[i] for i in pareto_indices]

        # Log Stats
        avg_loss = sum(f.validation_loss for f in fitnesses) / len(fitnesses)
        avg_vram = sum(f.vram_peak_mb for f in fitnesses) / len(fitnesses)
        logger.info(f"Stats: Avg Loss={avg_loss:.4f}, Avg VRAM={avg_vram:.1f}MB, Elites={len(elites)}")

        # 3. Save History
        self.history.append({
            "gen": generation,
            "pop_size": len(self.population),
            "avg_loss": avg_loss,
            "avg_vram": avg_vram,
            "pareto_size": len(elites)
        })

        # 4. Create Offspring
        next_pop = []

        num_elites_to_keep = min(len(elites), self.cfg['pop_size'] // 2)
        random.shuffle(elites)
        next_pop.extend(elites[:num_elites_to_keep])

        while len(next_pop) < self.cfg['pop_size']:
            # Select Parents
            p1 = self.select_tournament(self.population, fitnesses)
            p2 = self.select_tournament(self.population, fitnesses)

            # Crossover
            if random.random() < self.cfg['crossover_rate']:
                c1, c2 = crossover_uniform(p1, p2)
            else:
                c1, c2 = p1, p2
            
            # Mutation
            c1 = mutate_random_gene(c1, self.space, self.cfg['mutation_rate'])
            c2 = mutate_random_gene(c2, self.space, self.cfg['mutation_rate'])

            next_pop.append(c1)
            if len(next_pop) < self.cfg['pop_size']:
                next_pop.append(c2)
        self.population = next_pop

        # 5. Checkpoint
        if generation % self.cfg['checkpoint_freq'] == 0:
            self.save_checkpoint(generation)

    def run(self):
        """Execute the full search"""
        logger.info("Initialising population...")
        self.population = [self.space.sample() for _ in range(self.cfg['pop_size'])]

        for gen in range(1, self.cfg['n_generations'] + 1):
            self.step(gen)

        logger.info("Search Complete")
    
    def save_checkpoint(self, gen: int):
        filename = os.path.join(self.output_dir, f"checkpoint_gen_{gen}.json")
        data = {
            "generation": gen,
            "history": self.history,
            "population": [g.genes for g in self.population]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved checkpoint: {filename}")
        