import scipy.io
import numpy as np
import os
import pandas as pd

def list_datasets(path="datasets"):
    return [f for f in os.listdir(path) if f.endswith(".mat")]

def extract_capacity(cycles):
    cycle_nums = []
    capacities = []
    for i, cycle in enumerate(cycles):
        if cycle['type'][0] == 'charge':
            d = cycle['data'][0, 0]
            current = d['Current_charge'][0]
            time = d['Time'][0]
            try:
                time = [float(t) for t in time]
            except:
                time = time.astype(float)

            delta_t = np.diff(time)
            current_avg = (current[1:] + current[:-1]) / 2
            capacity = np.sum(current_avg * delta_t) / 3600
            cycle_nums.append(i + 1)
            capacities.append(capacity)
    return cycle_nums, capacities

def load_mat_data(filepath):
    data = scipy.io.loadmat(filepath)
    battery = data[list(data.keys())[-1]][0][0]  # assumes last key is the battery like 'B0005'
    cycles = battery['cycle'][0]

    cycle_nums = []
    capacities = []

    for i, cycle in enumerate(cycles):
        if cycle['type'][0] == 'charge':
            d = cycle['data'][0, 0]
            current = d['Current_charge'][0]
            time = d['Time'][0]
            try:
                time = [float(t) for t in time]
            except:
                time = time.astype(float)
            delta_t = np.diff(time)
            current_avg = (current[1:] + current[:-1]) / 2
            capacity = np.sum(current_avg * delta_t) / 3600
            cycle_nums.append(i + 1)
            capacities.append(capacity)

    df = pd.DataFrame({
        'Cycle': cycle_nums,
        'Capacity (Ah)': capacities
    })

    return df



