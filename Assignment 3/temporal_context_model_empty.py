from black import out
import numpy as np
from matplotlib import pyplot as plt
from numpy import random as rand
import random

N_WORLD_FEATURES = 5
N_ITEMS = 10
ENCODING_TIME = 500
TEST_TIME = 20

def draw_from_a_dist(p):
    if np.sum(p) == 0:
        p = 0.05 * np.ones(len(p))
    p /= np.sum(p)
    idx = np.where(rand.random(1) - np.cumsum(p) < 0)
    sample = np.min(idx)
    out = np.zeros(len(p))
    out[sample] = 1
    return out

schedule = np.column_stack((np.sort(np.round(rand.random(N_ITEMS) * ENCODING_TIME)), np.arange(N_ITEMS)))
schedule_load = ENCODING_TIME / np.median(np.diff(schedule[:, 0]))
encoding = np.zeros((N_ITEMS, N_WORLD_FEATURES + 1))

world_m = np.array([1, 2, 1, 2, 3], dtype=float)
world_var = 1
delta = 0.05
beta_param = 0.001
m = 0

for time in range(ENCODING_TIME):
    world_m += delta
    world = rand.normal(world_m, world_var)
    # 
    if m < N_ITEMS:
        if time == schedule[m, 0]:
            encoding[m, :] = np.append(world, m)
            m += 1

out = np.zeros(TEST_TIME)
while time < ENCODING_TIME + TEST_TIME:
    world_m += delta
    world = rand.normal(world_m, world_var)
    soa = np.zeros(N_ITEMS)
    for m in range(N_ITEMS):
        soa[m] = np.dot(encoding[m], np.append(world, m))
    soa /= np.linalg.norm(soa)
    out[time - ENCODING_TIME] = np.where(draw_from_a_dist(soa) > 0)[0]
    time += 1

success = len(np.unique(out))
