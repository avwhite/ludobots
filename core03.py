import numpy as np
from scipy.spatial.distance import euclidean
from itertools import product
import matplotlib.pyplot as plt
import random
from math import sqrt

def perturb_elem(e, prob):
    if prob > random.random():
        return random.random()*2 -1
    else:
        return e

perturb = np.vectorize(perturb_elem)

def simulate_neurons(neurons, synapses):
    time_steps, num_neurons = neurons.shape
    for t in range(1, time_steps):
        old_n_vals = neurons[t-1,:]
        for n in range(num_neurons):
            weights = synapses[n,:]
            temp = np.sum(old_n_vals * weights)
            if temp > 1:
                new_val = 1
            elif temp < 0:
                new_val = 0
            else:
                new_val = temp
            neurons[t,n] = new_val

def fitness(neuron_vals, desired_neuron_vals):
    last_n_vals = neuron_vals[-1,:]
    size, = last_n_vals.shape
    return 1 - euclidean(last_n_vals, desired_neuron_vals)/sqrt(size)
    #dist = last_n_vals - desired_neuron_vals
    #return 1 - sum(abs(dist))/10

def fitness2(neuron_vals):
    diff = 0
    for (i,j) in product(range(9), repeat=2):
        diff += abs(neuron_vals[i,j] - neuron_vals[i,j+1])
        diff += abs(neuron_vals[i,j] - neuron_vals[i+1,j])
    return diff/(2*9*9)

fig = plt.figure(figsize=(18,6))

generations = 1000
desired_neuron_vals = np.array([0,1,0,1,0,1,0,1,0,1])
parent_synapses = np.vectorize(lambda x: x*2 -1)(np.random.rand(10,10))
neuron_vals = np.empty([10,10])

neuron_vals[0,:] = 0.5
simulate_neurons(neuron_vals, parent_synapses)
parent_fitness = fitness(neuron_vals, desired_neuron_vals)
#parent_fitness = fitness2(neuron_vals)

ax1 = fig.add_subplot(1,3,1)
init_neurons = np.copy(neuron_vals)
ax1.imshow(init_neurons, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')

fitness_time = list()
fitness_time.append(parent_fitness)

for g in range(generations):
    child_synapses = perturb(parent_synapses, 0.05)
    neuron_vals[0,:] = 0.5
    simulate_neurons(neuron_vals, child_synapses)
    child_fitness = fitness(neuron_vals, desired_neuron_vals)
    #child_fitness = fitness2(neuron_vals)
    if child_fitness > parent_fitness:
        parent_synapses = child_synapses
        parent_fitness = child_fitness
    fitness_time.append(parent_fitness)

neuron_vals[0,:] = 0.5
simulate_neurons(neuron_vals, parent_synapses)


ax2 = fig.add_subplot(1,3,2)
ax2.imshow(neuron_vals, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')

ax3 = fig.add_subplot(1,3,3)
ax3.plot(fitness_time)

fig.savefig("out03.png", bbox_inches='tight')
