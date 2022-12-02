import numpy as np

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

max_fit = []
mean_fit = []

for gen in range(224):
    fitnesses = np.load('/Users/neythen/Desktop/Projects/worm_neural_networks/results/231122_evolution_constrained' + '/gen' + str(gen) + '/fitnesses.npy')
    population = np.load('/Users/neythen/Desktop/Projects/worm_neural_networks/results/231122_evolution_constrained' + '/gen' + str(gen) + '/population.npy')
    max_fit.append(np.max(fitnesses))
    mean_fit.append(np.mean(fitnesses))

order = np.argsort(fitnesses)[::-1]

fitnesses = fitnesses[order]
population = population[order]

print(fitnesses)
print(population[0:10])
plt.plot(max_fit)
plt.plot(mean_fit)

plt.figure()
plt.scatter(range(len(fitnesses)), fitnesses, marker='.')
plt.show()