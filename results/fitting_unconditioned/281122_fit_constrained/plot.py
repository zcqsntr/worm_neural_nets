import numpy as np
import os
import matplotlib.pyplot as plt

max_gen = 501
all_fitnesses = []
all_pops = []
for gen in range(max_gen+1):

    fitnesses = np.load('./gen' + str(gen) + '/fitnesses.npy')
    population = np.load('./gen' + str(gen) + '/population.npy')

    all_fitnesses.append(fitnesses)
    all_pops.append(population)

# find out how many adhere to the interactions we know about. Test against the weights we know are +ve and -ve and equivalent symettries
count = 0

for p in population:

    if np.all(np.array(p[:4]) >= 0) and np.all(np.array(p[4:]) <= 0):
        count += 1
    else:
        print(p)


print(count)

order = np.argsort(fitnesses)[::-1]
print(order)
fitnesses = np.array(fitnesses)[order]

population = population[order]

for i in range(len(population)):
    #print(fitnesses[i], population[i])
    pass


# load the test fitnesses (final population from evo algorithm tested using 1000 worms)

test_fitnesses = np.load('./fitnesses.npy')
test_population = np.load('./population.npy')

order = np.argsort(test_fitnesses)[::-1]
print(order)
test_fitnesses = np.array(test_fitnesses)[order]
test_population = np.array(test_population)[order]

print(test_population[0:10])

plt.plot(list(map(np.mean, all_fitnesses)), label = 'mean')
plt.plot(list(map(np.max, all_fitnesses)), label = 'max')
plt.legend()
plt.title('evolutionary alg')
plt.savefig('evolution.png', dpi= 300)

plt.figure()
plt.scatter(range(len(test_fitnesses)),test_fitnesses, marker='.', label = 'test')
plt.scatter(range(len(fitnesses)), fitnesses, marker='.', label = 'evo')
plt.title('Test results')
plt.legend()
plt.savefig('test_fitnesses.png', dpi= 300)

plt.figure()
plt.scatter(fitnesses, test_fitnesses)
plt.xlabel('fitnesses')
plt.ylabel('test fitnesses (more worms)')
plt.savefig('fitness_test_fitness.png', dpi= 300)
# test the 10 best worms


plt.show()