from dataclasses import dataclass
from .Scheduler import Scheduler
import random
import numpy as np

@dataclass
class Wolf:
    position: list[tuple[int, int]]
    score: tuple[float, float, float]

class GWOScheduler(Scheduler):
    def __init__(self, environment):
        super().__init__()
        self.setEnvironment(environment)
        self.num_wolves = 50
        self.num_step = 20

    def reset(self):
        self.hostMatrix = self.env.getHostMatrix()
        self.vmMatrix = self.env.getVmMatrix()
        self.global_best_score = (float('inf'), float('inf'), float('inf'))
        self.global_best_position = None

    def initial_wolves(self):
        wolves = []
        for _ in range(self.num_wolves):
            position = []
            for vmid in self.env.inactiveVmId:
                position.append((vmid, random.randint(0, len(self.hostMatrix)-1)))
            score = self.evaluate(position)
            wolves.append(Wolf(position=position, score=score))
        return wolves

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
        wolves = self.initial_wolves()

        for _ in range(self.num_step):
            wolves.sort(key=lambda w: (w.score[0], w.score[1], w.score[2]))
            alpha = wolves[0]
            beta = wolves[1]
            delta = wolves[2]

            a = 2 - (_ * (2/self.num_step))

            for w in wolves:
                new_position = []
                for i in range(len(self.vmMatrix)):
                    vmid = w.position[i][0]
                    
                    hosts = [alpha.position[i][1], beta.position[i][1], delta.position[i][1]]

                    host_count = {}
                    for h in hosts:
                        host_count[h] = host_count.get(h, 0) + 1
                        
                    max_votes = max(host_count.values())
                    candidates = [h for h, v in host_count.items() if v == max_votes]

                    new_hostid = random.choice(candidates)
                    new_position.append((vmid, new_hostid))
                new_score = self.evaluate(new_position)
                if self._is_better(new_score, w.score):
                    w.position = new_position
                    w.score = new_score
        wolves.sort(key=lambda w: (w.score[0], w.score[1], w.score[2]))
        best_wolf = wolves[0]
        
        return best_wolf.position
