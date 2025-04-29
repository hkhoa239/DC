from dataclasses import dataclass
import numpy as np
from .Scheduler import Scheduler
import random

@dataclass
class Gene:
    vmid: int
    hostid: int

class Chromesome:
    def __init__(self, genes):
        self.genes = genes

class GAScheduler(Scheduler):
    def __init__(self, environment):
        super().__init__()
        self.setEnvironment(environment)

        self.num_pop = 10
        self.num_gen = 5
        self.num_step = 20


    def update_best_solution(self, best_solve: Chromesome, best_score):
        self.best_solution = best_solve
        self.best_score = best_score
        self.best_vm_placement_matrix = [False] * len(self.env.inactiveVmId)

    def reset(self):
        self.num_pop = 10
        self.population = []
        self.best_chrom = None
        self.best_score = (float('inf'), float('inf'))
        self.hostMatrix = self.env.getHostMatrix()
        self.vmMatrix = self.env.getVmMatrix()

    def create_initial_population(self):
        population = []
        for i in range(self.num_pop):
            genes = []
            for j in range(len(self.vmMatrix)):
                hostid = random.randint(0, len(self.hostMatrix) - 1)
                genes.append(Gene(self.env.inactiveVmId[j], hostid))
            chrom = Chromesome(genes)
            population.append(chrom)
            self.evaluate(chrom)
            

        self.population = population
        # self.evaluate()

    def crossover(self):
        for _ in range(self.num_gen):
            index1 = random.sample(range(self.num_pop), 1)[0]
            index2 = random.sample(range(self.num_pop), 1)[0]
            while index1 == index2:
                index2 = random.sample(range(self.num_pop), 1)[0]
            chrom1 = self.population[index1]
            chomr2 = self.population[index2]

            crossover_point = random.randint(0, len(chrom1.genes) - 1)
            new_genes = chrom1.genes[:crossover_point] + chomr2.genes[crossover_point:]
            new_chom = Chromesome(new_genes)
            self.population.append(new_chom)
            self.num_pop += 1
            self.evaluate(new_chom)

    def mutation(self):
        for i in range(self.num_pop):
            if random.random() < 0.1:
                chrom = self.population[i]
                mutation_point = random.randint(0, len(chrom.genes) - 1)
                hostid = random.randint(0, len(self.hostMatrix) - 1)
                chrom.genes[mutation_point].hostid = hostid
                self.population[i] = chrom
                self.evaluate(chrom)

    def _is_better(self, score_a, score_b):
        return score_a[0] < score_b[0] and score_a[1] < score_b[1]

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

    def evaluate(self, chromesome):
        # score = 0
        matrix = self.hostMatrix
        # for chromesome in self.population:
        for gene in chromesome.genes:
            hostid = gene.hostid
            vmid = gene.vmid
            vm = self.env.getVmByID(vmid)
            matrix[hostid][2] += vm.core / matrix[hostid][0]
            matrix[hostid][3] += vm.ram / matrix[hostid][1]
            score = self.fitness(matrix)
            if self._is_better(score, self.best_score):
                self.best_score = score
                self.best_chrome = chromesome


    def run(self):
        self.reset()
        self.create_initial_population()
        for _ in range(self.num_step):
            self.crossover()
            self.mutation()

        return [(gene.vmid, gene.hostid) for gene in self.best_chrome.genes]