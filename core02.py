import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, cos, sin
from itertools import product

num_neurons=10
time_steps=50

neurons = np.zeros((time_steps,num_neurons))
for i in range(neurons.shape[1]):
    neurons[0,i] = random.random()

synapses = np.vectorize(lambda x: x*2 -1)(np.random.rand(num_neurons,num_neurons))

def draw_ann(ax, synapses):
    num_neurons = synapses.shape[0]
    neuron_positions = np.zeros((2,num_neurons))
    angle = 0.0
    angle_update = 2 * pi / num_neurons
    for i in range(num_neurons):
        neuron_positions[0,i] = sin(angle)
        neuron_positions[1,i] = cos(angle)
        angle += angle_update


    for (i,j) in product(range(num_neurons),repeat=2):
        (x1,y1) = neuron_positions[:,i]
        (x2,y2) = neuron_positions[:,j]
        w = synapses[i,j]
        color = [0,0,0] if w > 0 else [0.8,0.8,0.8]
        lw = int(10*abs(w)+1)
        ax.plot([x1,x2],[y1,y2], color=color, linewidth=lw)

    ax.plot(
        neuron_positions[0,:],
        neuron_positions[1,:],
        'ko',
        markerfacecolor=[1,1,1],
        markersize=18)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
draw_ann(ax1, synapses)
fig.savefig("out02.png")
