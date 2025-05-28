# Many Objectives Optimization (Developed base on NSGA-II) solve problem about EMO algorithms no more three
# objectives, so the performance is not good
# Ideal: solve individual objective one at a time and have prior

from dataclasses import dataclass
import numpy as np
from .Scheduler import Scheduler
from envs.Env import Env
import random

@dataclass
class Gene:
    vmid: int
    hostid: int

class Chromesome:
    def __init__(self, genes):
        self.genes = genes
        self.objectives = None
        self.normalized_objectives = None
        self.reference_point = None

class NSGAIIIScheduler(Scheduler):
    # Modeling Phase
    def __init__(self, environment):
        super().__init__()
        self.setEnvironment(environment)
        self.population_size = 10
        self.max_generations = 5
        self.num_objectives = 3
        # Tập hợp các điểm tham chiếu đều trên siêu phẳng trong không gian mục tiêu M chiều
        self.reference_points = self.generate_reference_points(self.num_objectives) # generate 3 reference points vertex (1.0.0), (0.1.0), (0.0.1)
        self.ideal_point = None
        self.extreme_points = None

    def reset(self):
        self.population_size = 10
        self.population = []
        self.best_chrom = None
        self.best_score = (float('inf'), float('inf'), float('inf'))
        self.hostMatrix = self.env.getHostMatrix()
        self.vmMatrix = self.env.getVmMatrix()
        
        
        self.ideal_point = None
        self.extreme_points = None

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = []
            for vmid in self.env.inactiveVmId:
                hostid = random.randint(0, len(self.env.hostlist) - 1)
                solution.append(Gene(vmid, hostid))
            chrom = Chromesome(solution)
            self.evaluate(chrom)  # Evaluate immediately after creation
            population.append(chrom)
        return population

    def generate_reference_points(self, num_objectives=3):
        H = 3 # Number of divisions
        ref_points = []
        # for i in range(H):
        #     for j in range(H):
        #         for k in range(H):
        #             ref_points.append((i/H, j/H, k/H))
        # return np.array(ref_points)
        def generate_points(remaining, depth, current):
            if depth == num_objectives - 1:
                current.append(remaining)
                ref_points.append(current.copy())
                current.pop()
                return
            for i in range(remaining + 1):
                current.append(i/H)
                generate_points(remaining - i, depth + 1, current)
                current.pop()
        generate_points(H, 0, [])
        return np.array(ref_points)
                
    # Evolution Phase
    def selection(self, fronts, population):
        if not fronts or not population:
            return []
            
        selected = []
        i = 0

        while i < len(fronts) and len(selected) + len(fronts[i]) <= self.population_size:
            selected.extend(fronts[i])
            i += 1

        if len(selected) < self.population_size and i < len(fronts):
            remaining = self.population_size - len(selected)
            last_front = fronts[i]

            ref_point_count = {i: 0 for i in range(len(self.reference_points))}

            for idx in selected:
                ref_point_count[population[idx].reference_point] += 1
                
            while remaining > 0 and last_front:
                min_count = min(ref_point_count.values())
                candidates = [idx for idx in last_front if ref_point_count[population[idx].reference_point] == min_count]

                if not candidates:
                    break

                selected_idx = random.choice(candidates)
                selected.append(selected_idx)
                last_front.remove(selected_idx)
                ref_point_count[population[selected_idx].reference_point] += 1
                remaining -= 1
        
        return [population[i] for i in selected]
    
    def crossover(self):
        for _ in range(self.max_generations):
            index1 = random.randint(0, len(self.population_size) - 1)
            index2 = random.randint(0, len(self.population_size) - 1)
            while index1 == index2:
                index2 = random.randint(0, len(self.population_size) - 1)
            parent1 = self.population[index1]
            parent2 = self.population[index2]

            crossover_point = random.randint(0, len(parent1.genes) - 1)
            new_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
            new_chromosome = Chromesome(new_genes)
            self.population.append(new_chromosome)
            self.population_size += 1
            self.evaluate(new_chromosome)
            
    def mutation(self):
        for i in range(self.population_size):
            if random.random() < 0.1:
                chromosome = self.population[i]
                mutation_point = random.randint(0, len(chromosome.genes) - 1)
                hostid = random.randint(0, len(self.hostMatrix) - 1)
                chromosome.genes[mutation_point].hostid = hostid
                self.population[i] = chromosome
                self.evaluate(chromosome)

    def evaluate(self, chromosome):
        if not chromosome.genes:
            chromosome.objectives = np.array([float('inf')] * self.num_objectives)
            return

        matrix = self.hostMatrix.copy()
        for gene in chromosome.genes:
            hostid = gene.hostid
            vmid = gene.vmid
            vm = self.env.getVmByID(vmid)
            matrix[hostid][2] += vm.core / matrix[hostid][0]
            matrix[hostid][3] += vm.ram / matrix[hostid][1]
        
        chromosome.objectives = self.fitness(matrix)

    def fitness(self, matrix):
        # Power consumption objective
        power_consumption = sum(host[0] * host[2] for host in matrix)
        
        # CPU imbalance objective
        total_core = sum(host[2] for host in matrix)
        avg_core = total_core / len(self.env.hostlist) if self.env.hostlist else 1
        cpu_imbalance = np.sqrt(sum((host[2]/avg_core - 1)**2 for host in matrix) / len(self.env.hostlist)) if self.env.hostlist else float('inf')
        
        # RAM imbalance objective
        total_ram = sum(host[3] for host in matrix)
        avg_ram = total_ram / len(self.env.hostlist) if self.env.hostlist else 1
        ram_imbalance = np.sqrt(sum((host[3]/avg_ram - 1)**2 for host in matrix) / len(self.env.hostlist)) if self.env.hostlist else float('inf')
        
        return np.array([power_consumption, cpu_imbalance, ram_imbalance])
    
    def non_dominated_sort(self, population):
        if not population:
            return [[]]
            
        fronts = [[]]
        domination_count = {i: 0 for i in range(len(population))}
        dominated_solutions = {i: set() for i in range(len(population))}
        
        # Calculate domination relationships
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if self.dominates(population[i].objectives, population[j].objectives):
                    dominated_solutions[i].add(j)
                    domination_count[j] += 1
                elif self.dominates(population[j].objectives, population[i].objectives):
                    dominated_solutions[j].add(i)
                    domination_count[i] += 1
        
        # Assign ranks
        for i in range(len(population)):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Generate fronts
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for j in fronts[i]:
                for k in dominated_solutions[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def normalize_objectives(self, population):
        """Normalize objectives using ideal point and extreme points"""
        if not population:
            return
            
        if self.ideal_point is None:
            self.ideal_point = np.min([p.objectives for p in population], axis=0)
        
        # Find extreme points
        if self.extreme_points is None:
            self.extreme_points = []
            for i in range(self.num_objectives):
                extreme_idx = np.argmin([p.objectives[i] for p in population])
                self.extreme_points.append(population[extreme_idx].objectives)
        
        # Normalize objectives
        for p in population:
            normalized = (p.objectives - self.ideal_point) / (self.extreme_points - self.ideal_point)
            p.normalized_objectives = normalized

    def associate_with_reference_points(self, population):
        """Associate solutions with reference points"""
        if not population:
            return
            
        for p in population:
            min_dist = float('inf')
            for i, ref_point in enumerate(self.reference_points):
                dist = np.linalg.norm(p.normalized_objectives - ref_point)
                if dist < min_dist:
                    min_dist = dist
                    p.reference_point = i

    
    def run(self):
        self.reset()
        population = self.initialize_population()
        
        for generation in range(self.max_generations):
            # Evaluate objectives for any unevaluated chromosomes
            for p in population:
                if p.objectives is None:
                    self.evaluate(p)

            # Normalize objectives
            self.normalize_objectives(population)
            
            # Associate with reference points
            self.associate_with_reference_points(population)

            # Non-dominated sorting
            fronts = self.non_dominated_sort(population)
            
            # Selection
            selected = self.selection(fronts, population)
            
            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                crossover_point = random.randint(0, len(parent1.genes) - 1)
                new_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
                child = Chromesome(new_genes)
                self.evaluate(child)  # Evaluate immediately after creation
                
                # Apply mutation
                if random.random() < 0.1:
                    mutation_point = random.randint(0, len(child.genes) - 1)
                    child.genes[mutation_point].hostid = random.randint(0, len(self.hostMatrix) - 1)
                    self.evaluate(child)  # Re-evaluate after mutation
                
                offspring.append(child)
            
            population = offspring

        # Select best solution from final population
        if not population:
            return []
            
        # Ensure all chromosomes are evaluated
        for p in population:
            if p.objectives is None:
                self.evaluate(p)
                
        best_idx = np.argmin([sum(p.objectives) for p in population])
        best_solution = population[best_idx]
        return [(gene.vmid, gene.hostid) for gene in best_solution.genes]
