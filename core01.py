import sys
import numpy as np
import matplotlib.pyplot as plt
import random

def perturb_elem(e, prob):
    if prob > random.random():
        return random.random()
    else:
        return e

def fitness_prog(genes, steps, prob):
    fits = np.empty(steps)
    gene_list = []
    perturb = np.vectorize(perturb_elem)

    parent = np.random.rand(genes)
    parent_fitness = np.mean(parent)

    for current_gen in range(steps):
        fits[current_gen] = parent_fitness
        gene_list.append(parent)

        child = perturb(parent, prob)
        child_fitness = np.mean(child)
        if child_fitness > parent_fitness:
            parent = child
            parent_fitness = child_fitness
    return (fits,np.vstack(gene_list))


def do_plots(filename):
    fig = plt.figure(figsize=(6,18))
    ax1 = fig.add_subplot(3,1,1)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.plot(fitness_prog(50,5000,0.05)[0])

    ax2 = fig.add_subplot(3,1,2)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    for i in range(5):
        ax2.plot(fitness_prog(50,5000,0.05)[0])

    ax3 = fig.add_subplot(3,1,3)
    ax3.set_xlabel("Gene")
    ax3.set_ylabel("Generation")
    (fits,genes) = fitness_prog(50,5000,0.05)
    ax3.imshow(genes, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    print(genes)

    fig.savefig(filename, bbox_inches='tight')

do_plots(sys.argv[1])
