from envs.Env import Env
from scheduler.MDQL import MDQLScheduler
import pandas as pd
import random
# def ga():
#     e = Env()
#     ga = GAScheduler(e)

#     data = {
#         "Interval": [],
#         "Power_Consumption": [],
#         "CPU_Utilization": [],
#         "Ram_Utilization": [],
#         "Inactivevmids": [],
#         "CPU_Imbalance": [],
#         "Ram_Imbalance": [],
#     }

#     state, info = e.reset()
#     for i in range(100):
#         print("Episode: ", i)
#         decision = ga.run()
#         _,rew, _, info = e.step(decision)

#         data["Interval"].append(info["Interval"])
#         data["Power_Consumption"].append(info["Power_Consumption"])
#         data["CPU_Utilization"].append(info["CPU_Utilization"])
#         data["Ram_Utilization"].append(info["Ram_Utilization"])
#         data["Inactivevmids"].append(info["Inactivevmids"])
#         data["CPU_Imbalance"].append(info["CPU_Imbalance"])
#         data["Ram_Imbalance"].append(info["Ram_Imbalance"])
#     path = "metrics/ga.csv"
#     df = pd.DataFrame(data=data)
#     df.to_csv(path,index=False)

# ga()

def train():
    env = Env()
    sche = MDQLScheduler(environment=env, checkpoint_path="")
    sche.train()

def run():
    e = Env()
    sche = MDQLScheduler(environment=e, checkpoint_path="dc4/train_ddql/2025-04-28T20-57-59/net_2.chkpt")


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
    for i in range(100):
        print("Episode: ", i)
        decision = sche.run()
        _,rew, _, info = e.step(decision)

        data["Interval"].append(info["Interval"])
        data["Power_Consumption"].append(info["Power_Consumption"])
        data["CPU_Utilization"].append(info["CPU_Utilization"])
        data["Ram_Utilization"].append(info["Ram_Utilization"])
        data["Inactivevmids"].append(info["Inactivevmids"])
        data["CPU_Imbalance"].append(info["CPU_Imbalance"])
        data["Ram_Imbalance"].append(info["Ram_Imbalance"])
    path = "metrics/mdql.csv"
    df = pd.DataFrame(data=data)
    df.to_csv(path,index=False)

run()