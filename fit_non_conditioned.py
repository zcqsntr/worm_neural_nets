
import os
import numpy as np
import sys
import time
from load_data import load_data
from wormSimulator import WormSimulator
import copy
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

def evolve_constraints(simulator, n_gens, pop_size, save_path = dir_path + '/working_dir/non_cond'):


    if fit_w8_w9:
        w_8 = np.random.random(size=(pop_size, 1)) * 10
        w_9 = np.random.random(size=(pop_size, 1)) * 10
    else:
        w_8 = np.ones(size=(pop_size, 1))
        w_9 = -np.ones(size=(pop_size, 1))

    w_signs = np.array([1,-1,-1,-1,-1,1,1,1,-1])

    population = np.hstack([np.random.random(size = (pop_size, 1))*10 *w for w in w_signs],  w_8,  w_9)


    fitnesses = simulator.get_fitnesses_par(population, n_worms)
    for i in range(n_gens):

        save_p = os.path.join(save_path, 'gen' + str(i))
        os.makedirs(save_p, exist_ok=True)

        np.save(save_p + '/population.npy', population)
        np.save(save_p + '/fitnesses.npy', fitnesses)

        order = np.argsort(fitnesses)[::-1]
        fitnesses = np.array(fitnesses)[order]

        population = population[order]



        population[int(pop_size * 0.4): int(pop_size * 0.8)] += np.random.random(size=(int(pop_size * 0.8) - int(pop_size * 0.4), 9)) * 2 - 1.

        # set weights to 0 if they break the sign constraints
        population[int(pop_size * 0.4): int(pop_size * 0.8), :][population[int(pop_size * 0.4): int(pop_size * 0.8), :] * w_signs < 0] = 0


        population[int(pop_size * 0.8):,:] = np.hstack([np.random.random(size = (pop_size - int(pop_size * 0.8), 1))*10*w for w in w_signs],  w_8,  w_9)




        if not fit_w8_w9:
            population[:, 3] = 1
            population[:, 8] = -1





        fitnesses[int(pop_size*0.4):] = simulator.get_fitnesses_par(population[int(pop_size*0.4):], n_worms)

        print('gen', i)
        print('max: ', np.max(fitnesses), population[0])
        print('mean: ', np.mean(fitnesses))

def evolve(simulator, n_gens, pop_size, save_path = './working_dir/aversive'):



    #population = np.load('/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/230111_evolution_constrained/gen99/population.npy')
    rand = np.random.random(size=(pop_size, 7)) * 20 - 10


    if fit_w8_w9:
        w_8 = np.random.random((pop_size, 1)) * 20 -10
        w_9 = np.random.random(size=(pop_size, 1)) * 20-10
    else:
        w_8 = np.ones((pop_size, 1))
        w_9 = -np.ones((pop_size, 1))

    population = np.hstack((rand, w_8,  w_9))

    fitnesses = simulator.get_fitnesses_par(population, n_worms)
    for i in range(n_gens):

        save_p = os.path.join(save_path, 'gen' + str(i))
        os.makedirs(save_p, exist_ok=True)

        np.save(save_p + '/population.npy', population)
        np.save(save_p + '/fitnesses.npy', fitnesses)

        order = np.argsort(fitnesses)[::-1]
        fitnesses = np.array(fitnesses)[order]

        population = population[order]


        #population[int(pop_size*0.5):] = population[:int(pop_size*0.5)] + np.random.random(size = ( int(pop_size*0.5), 9)) - 0.5

        #fitnesses[int(pop_size*0.5):] = simulator.get_fitnesses_par(population[int(pop_size*0.5):], n_worms)

        population[int(pop_size * 0.4): int(pop_size * 0.8)] += np.random.random(size=(int(pop_size * 0.8) - int(pop_size * 0.4), 9)) * 2 - 1.

        population[int(pop_size * 0.8):, :] = np.random.random(size=(pop_size - int(pop_size * 0.8), 9)) * 20 - 10

        if not fit_w8_w9:
            population[:, 3] = 1
            population[:, 8] = -1


        fitnesses[int(pop_size * 0.4):] = simulator.get_fitnesses_par(population[int(pop_size * 0.4):], n_worms)

        print('gen', i)
        print('max: ', np.max(fitnesses), population[0])
        print('mean: ', np.mean(fitnesses))

no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour = load_data(dir_path + '/data/behaviourdatabysector_NT.csv')

n_gens = 100
pop_size = 100


dataset = no_cond_odour
print(len(dataset))
n_worms = len(dataset)# number of worms in each experiment

n_worms = 2

w_1 = w_6 = w_7 = 1.5 # +ve weights
w_2 = w_3 = w_4 = w_5  = -1.5 # -ve weights
w_8 = 1
w_9 = -1



simulator = WormSimulator(dataset = dataset, dt = 0.005)
worm_trapped = False
conc_interval = None
fit_w8_w9 = True


opt = 'scan'


path = './results/worm_simulation_results_NT/230111_mock/fitting_output/'

if opt == 'E': # evolve
    evolve_constraints(simulator, n_gens, pop_size)
elif opt == 'P':  # plot
    t = time.time()
    population = np.load(path + 'weights_population.npy')
    fitnesses = np.load(path + 'final_fitnesses.npy')

    print(population.shape)

    order = np.argsort(fitnesses)[::-1]
    population = population[order]

    all_sectors = []

    all_sectors = simulator.run_experiment_par(population, n_worms)
    np.save(path + 'all_sectors.npy', all_sectors)
    ncols = 5
    fig, axs = plt.subplots(nrows=5, ncols=ncols, figsize=(15, 7.5))


    ms_errors = []
    ss_errors = []
    mr_errors = []
    sr_errors = []

    for i, sectors in enumerate(all_sectors):

        ax = axs[i // ncols, i  % ncols]


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
    print(time.time() - t)
    plt.show()

elif opt == 'scan':  # quick param scan after arantza's email
    starting_w1 = np.load(path + 'weights_population.npy')[2]
    starting_w2 = np.load(path + 'weights_population.npy')[15]


    all_test_weights = []

    for starting_weights in [starting_w1, starting_w2]:
        for w_1 in range(-10, 11, 2):

            for w_3 in range(-10, 11, 2):
                test_weights = copy.deepcopy(starting_weights)
                test_weights[0] = w_1
                test_weights[2] = w_3
                all_test_weights.append(test_weights)
    print(len(all_test_weights))
    all_sectors = simulator.run_experiment_par(all_test_weights, n_worms)
    np.save(path + 'param_scan/all_sectors.npy', all_sectors)

    worm_trapped = True
    conc_interval = [10, 40]
    max_t = 70
    calcium_sims = []

    for i in range(len(all_test_weights)):
        print(i)
        weights = all_test_weights[i]


        simulator.t_span[-1] = max_t
        solution = simulator.forward_euler(simulator.y0, weights)
        calcium_sims.append(solution)
    np.save(path + 'param_scan/calcium_sims.npy', calcium_sims)



elif opt == 'S': # simulate
    population = np.load(
        path + '/new_weights_population.npy')

    fitnesses = np.load(path + '/final_fitnesses.npy')

    order = np.argsort(fitnesses)[::-1]
    population = population[order]

    weights = copy.deepcopy(population[0])

    sol = simulator.forward_euler(simulator.y0, weights)

    print(simulator.score_worm(sol))

    simulator.plot_sol(sol)

    plt.show()

elif opt == 'C': # test worm in the calcium imaging experiment
    worm_trapped = True
    conc_interval = [10, 40]
    max_t = 70
    params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
              speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, worm_trapped, conc_interval]

    #population = np.load(path + 'gen99/population.npy')

    population = np.load(path + 'weights_population.npy')
    ncols = 10
    fig, axs = plt.subplots(nrows=10, ncols=ncols, figsize=(15, 7.5))
    calcium_sims = []

    for i in range(len(population)):
        weights = population[i]

        # positive weights
        params[10] = weights[0]
        params[15] = weights[1]
        params[16] = weights[2]
        params[17] = weights[3]

        # negative weights
        params[11] = weights[4]
        params[12] = weights[5]
        params[13] = weights[6]
        params[14] = weights[7]
        params[18] = weights[8]

        simulator.t_span[-1] = max_t
        solution = simulator.forward_euler(simulator.y0, params)
        calcium_sims.append(solution)

        # plot neuron voltages

        ax = axs[i // ncols, i  % ncols]
        ax.set_ylim(bottom=-1., top = 1.)
        ax.plot(np.arange(0, max_t, simulator.dt), solution[0, 1:-1], label='AWC')
        ax.plot(np.arange(0, max_t, simulator.dt), solution[3, 1:-1], label='AIB')
        ax.plot(np.arange(0, max_t, simulator.dt), solution[5, 1:-1], label='AIY')
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron voltages')


        #simulator.plot_conc()



    #np.save('/results/final_results/230111_mock/fitting_data/final_calcium_sims.npy', calcium_sims)
    plt.show()