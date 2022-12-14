
import os
import numpy as np
import sys
import time
from load_data import load_data
from wormSimulator import WormSimulator

on_cluster = len(sys.argv) > 1
dir_path = os.path.dirname(os.path.realpath(__file__))

if not on_cluster:
    import matplotlib as mpl

    mpl.use('tkagg')
    import matplotlib.pyplot as plt



def param_scan(simulator, start, stop, step, save_path = dir_path + '/working_dir/param_scan', plot=True):
    os.makedirs(save_path, exist_ok = True)
    population = np.arange(start, stop, step).reshape(-1, 1)

    fitnesses, all_sectors  = simulator.get_fitnesses(population)



    if plot:
        plt.plot(population, fitnesses)


    np.save(save_path + '/params.npy',population)
    np.save(save_path + '/fitnesses.npy',fitnesses)
    np.save(save_path + '/sectors.npy', all_sectors)

    if plot:
        plt.show()

def evolve_constraints(simulator, n_gens, pop_size, save_path = dir_path + '/working_dir/evolution_constrained'):

    pos = np.random.random(size = (pop_size, 3))*10 # population of positive weights
    neg = np.random.random(size = (pop_size, 4))*-10  # population of negative weights

    population = np.hstack((pos, neg))


    fitnesses = simulator.get_fitnesses_par(population, n_worms)
    for i in range(n_gens):

        save_p = os.path.join(save_path, 'gen' + str(i))
        os.makedirs(save_p, exist_ok=True)

        np.save(save_p + '/population.npy', population)
        np.save(save_p + '/fitnesses.npy', fitnesses)

        order = np.argsort(fitnesses)[::-1]
        fitnesses = np.array(fitnesses)[order]

        population = population[order]


        population[int(pop_size*0.4): int(pop_size*0.8)] += np.random.random(size = (int(pop_size*0.8)- int(pop_size*0.4), 7))*2 - 1.
        population[int(pop_size*0.4): int(pop_size*0.8), 0:3][population[int(pop_size*0.4): int(pop_size*0.8), 0:3] < 0] = 0
        population[int(pop_size*0.4): int(pop_size*0.8), 3:7][population[int(pop_size*0.4): int(pop_size*0.8), 3:7] > 0] = 0


        population[int(pop_size*0.8): , 0:3] = np.random.random(size = (pop_size-int(pop_size*0.8), 3))*10
        population[int(pop_size*0.8): , 3:7] = np.random.random(size = (pop_size-int(pop_size*0.8), 4))*-10

        fitnesses[int(pop_size*0.4):] = simulator.get_fitnesses_par(population[int(pop_size*0.4):], n_worms)

        print('gen', i)
        print('max: ', np.max(fitnesses), population[0])
        print('mean: ', np.mean(fitnesses))


no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour = load_data(dir_path + '/data/behaviourdatabysector_NT.csv')

n_gens = 100
pop_size = 100


dataset = no_cond_odour
print(len(dataset))
n_worms = len(dataset)# number of worms in each experiment


# starting params from gosh et al
tm = 0.5 #s
AIB_v0 = AIA_v0 = AIY_v0 = AWC_v0 = 0
AWC_gain = 2
AWC_f_a = 4 #1/s
AWC_f_b = 15 #1/s
AWC_s_gamma = 2 #1/s
speed = 0.11 #mm/s


w_1 = w_6 = w_7 = 1.5 # +ve weights
w_2 = w_3 = w_4 = w_5  = -1.5 # -ve weights



simulator = WormSimulator(dataset = dataset, dt = 0.1)
worm_trapped = False
conc_interval = None

params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
          speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, worm_trapped, conc_interval]

#sol = solve_ivp(xdot, t_span, y0, t_eval = np.arange(t_span[-1]), args = (p,)).y

opt = 'E'







path = '/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/221214_fit_constrained/'

if opt == 'E': # evolve
    evolve_constraints(simulator, n_gens, pop_size)
elif opt == 'P':  # plot
    population = np.load(path + 'population.npy')
    fitnesses = np.load(path + 'fitnesses.npy')

    print(population[0:10])

    order = np.argsort(fitnesses)[::-1]
    population = population[order]

    all_sectors = []

    all_sectors = simulator.run_experiment_par(population[0:25], n_worms)

    ncols = 5
    fig, axs = plt.subplots(nrows=5, ncols=ncols, figsize=(15, 7.5))


    ms_errors = []
    ss_errors = []
    mr_errors = []
    sr_errors = []

    for i, sectors in enumerate(all_sectors):

        ax = axs[i // ncols, i  % ncols]
        ax.violinplot([sum(s) for s in sectors])

        ax.set_ylim(bottom=-6.1, top=6.1)

        ax.violinplot(list(map(sum, dataset)))
        ax.violinplot(list(map(sum, sectors)))



    fig.suptitle('Violin plots of top 25 members of the evolutionary population')
    plt.savefig('violin_plots.png', dpi=300)

    print('score', np.mean(list(map(sum, dataset))), 'score std', np.std(list(map(sum, dataset))), 'range',
          np.mean(list(map(lambda x: max(x) - min(x), dataset))), 'range std',
          np.std(list(map(lambda x: max(x) - min(x), dataset))))


    print('score', np.mean(list(map(sum, sectors))), 'score std', np.std(list(map(sum, sectors))), 'range',
          np.mean(list(map(lambda x: max(x) - min(x), sectors))), 'range std',
          np.std(list(map(lambda x: max(x) - min(x), sectors))))

    plt.show()

elif opt == 'T': # test
    n_worms = 1000

    population = np.load(path + 'gen144/population.npy')
    fitnesses = np.load(path + 'gen144/fitnesses.npy')
    order = np.argsort(fitnesses)[::-1]
    print(order)

    t = time.time()
    fitnesses = simulator.get_fitnesses_par(population, n_worms)
    print(time.time() - t)
    order = np.argsort(fitnesses)[::-1]
    print(order)

    np.save(path + 'population.npy', population)
    np.save(path + 'fitnesses.npy', fitnesses)

elif opt == 'S': # simulate
    population = np.load(
        path + 'population.npy')

    fitnesses = np.load(path + 'fitnesses.npy')

    order = np.argsort(fitnesses)[::-1]
    population = population[order]

    p = population[0]
    print(p)


    # positive weights
    params[10] = p[0]
    params[15] = p[1]
    params[16] = p[2]

    # negative weights
    params[11] = p[4]
    params[12] = p[5]
    params[13] = p[6]
    params[14] = p[7]

    sol = simulator.forward_euler(simulator.y0, params)

    print(simulator.score_worm(sol))

    simulator.plot_sol(sol)

    plt.show()

elif opt == 'C': # test worm in the calcium imaging experiment
    worm_trapped = True
    conc_interval = [10, 40]
    max_t = 70
    params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
              speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, worm_trapped, conc_interval]

    population = np.load(
        path + 'population.npy')
    ncols = 5
    fig, axs = plt.subplots(nrows=5, ncols=ncols, figsize=(15, 7.5))
    for i in range(25):
        p = population[i]

        # positive weights
        params[10] = p[0]
        params[15] = p[1]
        params[16] = p[2]

        # negative weights
        params[11] = p[4]
        params[12] = p[5]
        params[13] = p[6]
        params[14] = p[7]

        simulator.t_span[-1] = max_t
        solution = simulator.forward_euler(simulator.y0, params)

        # plot neuron voltages
        ax = axs[i // ncols, i  % ncols]
        ax.plot(np.arange(0, max_t, simulator.dt), solution[0, 1:-1], label='AWC')
        ax.plot(np.arange(0, max_t, simulator.dt), solution[3, 1:-1], label='AIB')
        ax.plot(np.arange(0, max_t, simulator.dt), solution[5, 1:-1], label='AIY')
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron voltages')
        #simulator.plot_conc()
    plt.show()