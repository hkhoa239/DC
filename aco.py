from envs.Env import Env
from scheduler.ACO import ACOScheduler
import pandas as pd
import os

def aco():
    e = Env()
    aco = ACOScheduler(e)

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
        print("\rEpisode: ", i, end = '')
        decision = aco.run()
        _,rew, _, info = e.step(decision)

        data["Interval"].append(info["Interval"])
        data["Power_Consumption"].append(info["Power_Consumption"])
        data["CPU_Utilization"].append(info["CPU_Utilization"])
        data["Ram_Utilization"].append(info["Ram_Utilization"])
        data["Inactivevmids"].append(info["Inactivevmids"])
        data["CPU_Imbalance"].append(info["CPU_Imbalance"])
        data["Ram_Imbalance"].append(info["Ram_Imbalance"])
    path = os.path.join(os.path.curdir, "metrics", "aco.csv")
    print (f"\nSaving metrics to {os.path.abspath(path)}. Is path exist? {os.path.exists(path)}")
    df = pd.DataFrame(data=data)
    df.to_csv(path,index=False)

aco()