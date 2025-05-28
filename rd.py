from envs.Env import Env
from scheduler.RD import RDScheduler
import pandas as pd
import random

def rf():
    e = Env()
    ga = RDScheduler(e)

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
    for i in range(1000):
        print("Episode: ", i)
        decision = ga.run()
        _,rew, _, info = e.step(decision)

        data["Interval"].append(info["Interval"])
        data["Power_Consumption"].append(info["Power_Consumption"])
        data["CPU_Utilization"].append(info["CPU_Utilization"])
        data["Ram_Utilization"].append(info["Ram_Utilization"])
        data["Inactivevmids"].append(info["Inactivevmids"])
        data["CPU_Imbalance"].append(info["CPU_Imbalance"])
        data["Ram_Imbalance"].append(info["Ram_Imbalance"])
    path = "metrics/rd.csv"
    df = pd.DataFrame(data=data)
    df.to_csv(path,index=False)

rf()