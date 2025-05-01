from dataclasses import dataclass
import numpy as np
import random
from .Scheduler import Scheduler

@dataclass
class Gene:
    vmid: int
    hostid: int

class Chromesome:
    def __init__(self, genes):
        self.genes = genes

class ACOScheduler(Scheduler):
    def __init__(self, environment):
        super().__init__()
        self.setEnvironment(environment)
        self.num_ants = 10
        self.num_iterations = 20
        self.alpha = 1  # pheromone influence
        self.beta = 2   # heuristic influence
        self.evaporation_rate = 0.5
        self.pheromone_init = 1.0

    def reset(self):
        self.best_score = (float('inf'), float('inf'))
        self.best_solution = None
        self.vmMatrix = self.env.getVmMatrix()
        self.hostMatrix = self.env.getHostMatrix()
        self.pheromone = np.full((len(self.vmMatrix), len(self.hostMatrix)), self.pheromone_init)

    def fitness(self, matrix):
        totalCore = 0
        for host in matrix:
            totalCore += host[2]
        avgCore = totalCore / len(self.env.hostlist)
        imCore = 0
        for host in matrix:
            imCore += (host[2]/avgCore-1)**2 if avgCore != 0 else 0
        imCore = np.sqrt(imCore / len(self.env.hostlist))

        totalRam = 0
        for host in matrix:
            totalRam += host[3]
        avgRam = totalRam / len(self.env.hostlist)
        imRam = 0
        for host in matrix:
            imRam += (host[3]/avgRam - 1)**2 if avgRam != 0 else 0
        imRam = np.sqrt(imRam / len(self.env.hostlist))

        return (imCore, imRam)

    def _is_better(self, score_a, score_b):
        return score_a[0] < score_b[0] and score_a[1] < score_b[1]

    def construct_solution(self):
        genes = []
        tempMatrix = [row.copy() for row in self.env.getHostMatrix()]
        for i, _ in enumerate(self.vmMatrix):
            probabilities = []
            for j, host in enumerate(tempMatrix):
                heuristic = 1.0 / ((host[2]+1)*(host[3]+1))  # Inverse of utilization
                prob = (self.pheromone[i][j]**self.alpha) * (heuristic**self.beta)
                probabilities.append(prob)
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]

            hostid = random.choices(range(len(tempMatrix)), weights=probabilities)[0]
            gen = Gene(self.env.inactiveVmId[i], hostid)
            genes.append(gen)

            vm = self.env.getVmByID(gen.vmid)
            tempMatrix[hostid][2] += vm.core / tempMatrix[hostid][0]  # CPU utilization
            tempMatrix[hostid][3] += vm.ram / tempMatrix[hostid][1]   # RAM utilization

        return Chromesome(genes), tempMatrix

    def update_pheromones(self, solutions):
        self.pheromone *= (1 - self.evaporation_rate)
        for chrom, matrix in solutions:
            score = self.fitness(matrix)
            if self._is_better(score, self.best_score):
                self.best_score = score
                self.best_solution = chrom
            for i, gene in enumerate(chrom.genes):
                self.pheromone[i][gene.hostid] += 1.0 / (score[0] + score[1] + 1e-6)  # small epsilon to avoid division by 0

    def run(self):
        self.reset()
        for _ in range(self.num_iterations):
            solutions = []
            for _ in range(self.num_ants):
                chrom, matrix = self.construct_solution()
                solutions.append((chrom, matrix))
            self.update_pheromones(solutions)

        return [(gene.vmid, gene.hostid) for gene in self.best_solution.genes]
