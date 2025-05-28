from envs.Env import Env
from scheduler.PPO import PPO
import pandas as pd
import random
def pso():
    e = Env()
    pso = PSOScheduler(e)

    data = {
        "Interval": [],
        "Power_Consumption": [],
        "CPU_Utilization": [],
        "Ram_Utilization": [],
        "Inactivevmids": [],
        "CPU_Imbalance": [],
        "Ram_Imbalance": [],
    }

    state, info = e.reset()
    # decision = pso.run()
    # print("Decision: ", decision)

    for i in range(1000):
        print("Episode: ", i)
        decision = pso.run()
        _,rew, _, info = e.step(decision)

        data["Interval"].append(info["Interval"])
        data["Power_Consumption"].append(info["Power_Consumption"])
        data["CPU_Utilization"].append(info["CPU_Utilization"])
        data["Ram_Utilization"].append(info["Ram_Utilization"])
        data["Inactivevmids"].append(info["Inactivevmids"])
        data["CPU_Imbalance"].append(info["CPU_Imbalance"])
        data["Ram_Imbalance"].append(info["Ram_Imbalance"])
    path = "metrics/pso.csv"
    df = pd.DataFrame(data=data)
    df.to_csv(path,index=False)

pso()