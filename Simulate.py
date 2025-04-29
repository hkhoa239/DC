from envs.Env import Env

e = Env()
state, info = e.reset()
print(info)
print("------------------------------------------------")
state, rew, done, info = e.step([(0,0)])
print(info)
print("------------------------------------------------")

state, rew, done, info = e.step([(1,0)])
print(info)
print("------------------------------------------------")

state, rew, done, info = e.step([(2,1)])
print(info)
print("------------------------------------------------")
state, rew, done, info = e.step([(3,4),(5,6)])
print(info)
print("------------------------------------------------")
state, rew, done, info = e.step([(4,1)])
print(info)
print("------------------------------------------------")
