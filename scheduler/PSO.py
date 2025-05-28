from dataclasses import dataclass
from .Scheduler import Scheduler
import random
import numpy as np

@dataclass
class Particle:
    position: list[int]
    velectory: list[float]
    personal_best_position: list[tuple[int, int]]
    personal_best_score: tuple[float, float, float]

class PSOScheduler(Scheduler):
    def __init__(self, environment):
        super().__init__()
        self.setEnvironment(environment)
        self.num_particles = 50
        self.num_step = 20


    def reset(self):
        self.hostMatrix = self.env.getHostMatrix()
        self.vmMatrix = self.env.getVmMatrix()
        self.global_best_score = (float('inf'), float('inf'), float('inf'))
        self.global_best_position = None


    def initialize_particles(self):
        particles = []

        for _ in range(self.num_particles):
            position = []
            velectory = [random.uniform(-1, 1) for _ in range(len(self.vmMatrix))]
            for vmid in self.env.inactiveVmId:
                position.append((vmid, random.randint(0, len(self.hostMatrix) - 1)))
            
            score = self.evaluate(position)
            particles.append(
                Particle(
                    position=position,
                    velectory=velectory,
                    personal_best_position=position,
                    personal_best_score=score
                )
            )
            if self._is_better(score, self.global_best_score):
                self.global_best_position = position
                self.global_best_score = score

        return particles


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

        powerCore = 0
        for host in matrix:
            powerCore += host[0] * host[2]

        return (imCore, imRam, powerCore)

    def _is_better(self, score1, score2):
        return score1[0] < score2[0] and score1[1] < score2[1] and score1[2] < score2[2]

    def evaluate(self, position):
        matrix = self.hostMatrix
        for vmid, hostid in position:
            vm = self.env.getVmByID(vmid)
            matrix[hostid][2] += vm.core / matrix[hostid][0]
            matrix[hostid][3] += vm.ram / matrix[hostid][1]
            score = self.fitness(matrix)
        return score

    def run(self):
        self.reset()
        particles = self.initialize_particles()
        w = 0.5
        c1 = 1.5
        c2 = 1.5

        for _ in range(self.num_step):
            for p in particles:
                for i in range(len(self.vmMatrix)):
                    r1 = random.random()
                    r2 = random.random()

                    p.velectory[i] = (
                        w * p.velectory[i] + c1 * r1 * int(p.personal_best_position[i] != p.position[i]) + c2 * r2 * int(self.global_best_position[i] != p.position[i])
                    )
                    vmid = p.position[i][0]
                    cur_hostid = p.position[i][1]
                    pbest_hostid = p.personal_best_position[i][1]
                    gbest_hostid = self.global_best_position[i][1]

                    candidates = []
                    if pbest_hostid != cur_hostid:
                        candidates.append((pbest_hostid, c1))
                    if gbest_hostid != cur_hostid:
                        candidates.append((gbest_hostid, c2))

                    if candidates:
                        total_weight = sum(w for _, w in candidates)
                        r = random.uniform(0, total_weight)
                        cum_weight = 0
                        for hostid, w in candidates:
                            cum_weight += w
                            if r <= cum_weight:
                                p.position[i] = (vmid, hostid)
                                break
                    else:
                        if random.random() < min(1, abs(p.velectory[i])):
                            candidate_hostids = [hostid for hostid in range(len(self.hostMatrix)) if hostid != cur_hostid]
                            if candidate_hostids:
                                r = random.choice(candidate_hostids)
                                p.position[i] = (vmid, r)
                score = self.evaluate(p.position)
                if self._is_better(score, p.personal_best_score):
                    p.personal_best_position = p.position[:]
                    p.personal_best_score = score
                if self._is_better(score, self.global_best_score):
                    self.global_best_position = p.position[:]
                    self.global_best_score = score

        return self.global_best_position
