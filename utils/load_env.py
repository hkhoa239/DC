import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from envs.Env import Env

def load_env(params=None):
    if params:
        return Env(TotalPower=params["TotalPower"], RouterBw=params["RouterBw"], HostLimit=params["HostLimit"], ContainerLimit=params["ContainerLimit"], IntervalTime=params["IntervalTime"], meanJ=params["meanJ"], sigmaJ=params["sigmaJ"])
    return Env()