from envs.Env import Env
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import json
def simulate():
    TotalPower=2000
    RouterBw=10000
    ContainerLimit=5000
    IntervalTime=1
    HostLimit=10
    e = Env(TotalPower=TotalPower, RouterBw=RouterBw, ContainerLimit=ContainerLimit, IntervalTime=IntervalTime, HostLimit=HostLimit, meanJ=10, sigmaJ=2)
    # state, info = e.reset()
    # print(info)
    # print("------------------------------------------------------------------------------------------")
    # state, reward, done, info = e.step(e.filter_action(1))
    # print(info)
    # print("reward: ",reward)
    # print("------------------------------------------------------------------------------------------")
    # state, reward, done, info = e.step(e.filter_action(0))
    # print(info)
    # print("reward: ",reward)
    # print("------------------------------------------------------------------------------------------")
    # state, reward, done, info = e.step(e.filter_action(2))
    # print(info)
    # print("reward: ",reward)
    # print("------------------------------------------------------------------------------------------")
    # state, reward, done, info = e.step(e.filter_action(1))
    # print(info)
    # print("reward: ",reward)
    # print("------------------------------------------------------------------------------------------")
    # state, reward, done, info = e.step(e.filter_action(1))
    # print(info)
    # print("reward: ",reward)
    # print("------------------------------------------------------------------------------------------")
    # state, reward, done, info = e.step(e.filter_action(2))
    # print(info)
    # print("reward: ",reward)
    # print("------------------------------------------------------------------------------------------")
    # state, reward, done, info = e.step(e.filter_action(1))
    # print(info)
    # print("reward: ",reward)
    # print("------------------------------------------------------------------------------------------")
    e.reset()
    step = 0
    while (step < 1000):
        action = random.randint(0,HostLimit-1)
        e.step(e.filter_action(action))
        print(step)
        step += 1

def save_plot(fig, filename):
    fig.savefig(filename)

def plot_power_consumption_from_json(json_path, save_path="simdata/host_power.png"):
    # Load data from JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract power consumption over steps for each host
    plt.figure(figsize=(8, 5))
    for host_id in set(host['id'] for step in data['random_data'] for host in step['hosts']):
        step_counts = []
        power_values = []
        for step in data['random_data']:
            for host in step['hosts']:
                if host['id'] == host_id:
                    step_counts.append(step['step'])
                    power_values.append(host['power'])
        plt.plot(step_counts, power_values, marker='o', linestyle='-', label=f'Host {host_id}')
    
    plt.xlabel('Step Count')
    plt.ylabel('Power Consumption')
    plt.title('Power Consumption Over Steps')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)

def plot_avg_response_time_from_json(json_path, save_path="simdata/response_time.png"):
    # Load data from JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract step count and average response time
    step_counts = []
    avg_response_times = []
    for step in data['random_data']:
        if step['respone_time_history']:
            avg_response_time = sum(step['respone_time_history']) / len(step['respone_time_history'])
        else:
            avg_response_time = 0
        step_counts.append(step['step'])
        avg_response_times.append(avg_response_time)

    # Plot data
    plt.figure(figsize=(8, 5))
    plt.plot(step_counts, avg_response_times, marker='o', linestyle='-', color='r', label='Avg Response Time')
    plt.xlabel('Step Count')
    plt.ylabel('Average Response Time')
    plt.title('Average Response Time Over Steps')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)

def plot_fifo_length_from_json(json_path, save_path="simdata/fifo_length.png"):
    # Load data from JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract step count and FIFO length
    step_counts = []
    fifo_lengths = []
    for step in data['random_data']:
        step_counts.append(step['step'])
        # print(len(step['fifo']))
        fifo_lengths.append(step['fifo'])
    
    # Plot data
    plt.figure(figsize=(8, 5))
    plt.plot(step_counts, fifo_lengths, marker='o', linestyle='-', color='g', label='FIFO Length')
    plt.xlabel('Step Count')
    plt.ylabel('FIFO Length')
    plt.title('FIFO Length Over Steps')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)

def plot_utilization_from_json(json_path, save_path="simdata/host_utilization.png"):
    # Load data from JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract utilization over steps for each host
    plt.figure(figsize=(8, 5))
    host_id = 0
    for host_id in set(host['id'] for step in data['random_data'] for host in step['hosts']):
        step_counts = []
        utilization_values = []
        for step in data['random_data'][450:500]:
            for host in step['hosts']:
                if host['id'] == host_id:
                    step_counts.append(step['step'])
                    utilization_values.append(host['utilization'])
        plt.plot(step_counts, utilization_values, marker='o', linestyle='-', label=f'Host {host_id}')
    
    plt.xlabel('Step Count')
    plt.ylabel('Utilization')
    plt.title('Utilization Over Steps')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)

def plot_datacenter_power_consumption_from_json(json_path, save_path="simdata/dc_total_power.png"):
    # Load data from JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract step count and total power consumption per step
    step_counts = []
    total_power_consumptions = []
    for step in data['random_data']:
        step_counts.append(step['step'])
        total_power_consumptions.append(step['power_consumption'])
    
    # Plot data
    plt.figure(figsize=(8, 5))
    plt.plot(step_counts, total_power_consumptions, marker='o', linestyle='-', color='b', label='DataCenter Power Consumption')
    plt.xlabel('Step Count')
    plt.ylabel('Total Power Consumption')
    plt.title('DataCenter Power Consumption Over Steps')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)


def plot_completed_from_json(json_path, save_path="simdata/completed.png"):
    # Load data from JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract step count and total power consumption per step
    step_counts = []
    completed = []
    for step in data['random_data']:
        step_counts.append(step['step'])
        completed.append(step['completed'])
    
    # Plot data
    plt.figure(figsize=(8, 5))
    plt.plot(step_counts, completed, marker='o', linestyle='-', color='b', label='Completed Over Step')
    plt.xlabel('Step Count')
    plt.ylabel('Completed')
    plt.title('Completed Over Step')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)


def plot_dropped_from_json(json_path, save_path="simdata/dropped.png"):
    # Load data from JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract step count and total power consumption per step
    step_counts = []
    dropped = []
    for step in data['random_data']:
        step_counts.append(step['step'])
        dropped.append(step['dropped'])
    
    # Plot data
    plt.figure(figsize=(8, 5))
    plt.plot(step_counts, dropped, marker='o', linestyle='-', color='b', label='Dropped Over Step')
    plt.xlabel('Step Count')
    plt.ylabel('Dropped')
    plt.title('Dropped Over Step')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)

def plot_rew(json_path="simdata/data.json", save_path="simdata/reward.png"):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract step count and average response time
    step_counts = []
    rewards = []
    for step in data['random_data']:
        if step['respone_time_history']:
            avg_response_time = sum(step['respone_time_history']) / len(step['respone_time_history'])
        else:
            avg_response_time = 0
        step_counts.append(step['step'])
        r_r = avg_response_time/max(step['respone_time_history']) if len(step['respone_time_history']) > 0 else 0
        r_e = step['power_consumption'] / 2000
        r = -0.5*r_r + -0.5*r_e + -step['dropped'] / step['completed']
        rewards.append(r)
    plt.figure(figsize=(8, 5))
    plt.plot(step_counts, rewards, marker='o', linestyle='-', color='b', label='Reward Over Step')
    plt.xlabel('Step Count')
    plt.ylabel('Reward')
    plt.title('Reward Over Step')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()
    save_plot(fig, save_path)


# simulate()
def plot():
    # plot_avg_response_time_from_json(json_path="simdata/data.json")
    # plot_datacenter_power_consumption_from_json(json_path="simdata/data.json")
    plot_utilization_from_json(json_path="simdata/data.json")
    # plot_power_consumption_from_json(json_path="simdata/data.json")
    # plot_completed_from_json(json_path="simdata/data.json")
    # plot_dropped_from_json(json_path="simdata/data.json")
    # plot_rew()
plot()

