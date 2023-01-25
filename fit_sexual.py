
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import os
import time
from load_data import load_data
from wormSimulator import WormSimulator







def param_scan(simulator, start, stop, step, save_path = './working_dir/param_scan', plot=True):
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



def evolve_constraints(simulator, n_gens, pop_size, save_path = './working_dir/sexual'):



    #population = np.load('/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_aversive/221216_evolution_constrained/gen192/population.npy')
    pos = np.random.random(size=(pop_size, 3)) * 20 - 10
    neg = np.random.random(size=(pop_size, 4)) * 20 - 10

    if fit_w8_w9:
        w_8 = np.random.random(size=(pop_size, 1)) * 20 - 10
        w_9 = np.random.random(size=(pop_size, 1)) * 20 - 10
    else:
        w_8 = np.ones((pop_size, 1))
        w_9 = -np.ones((pop_size, 1))

    population = np.hstack((pos, w_8, neg, w_9))

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

        population[int(pop_size * 0.4): int(pop_size * 0.8)] += np.random.random(
            size=(int(pop_size * 0.8) - int(pop_size * 0.4), 9)) * 2 - 1.

        population[int(pop_size * 0.8):, :] = np.random.random(size=(pop_size - int(pop_size * 0.8), 9)) * 20 - 10

        if not fit_w8_w9:
            population[:, 3] = 1
            population[:, 8] = -1

        fitnesses[int(pop_size * 0.4):] = simulator.get_fitnesses_par(population[int(pop_size * 0.4):], n_worms)

        print('gen', i)
        print('max: ', np.max(fitnesses), population[0])
        print('mean: ', np.mean(fitnesses))


no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour = load_data('./data/behaviourdatabysector_NT.csv')
print([len(x) for x in [no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour]])
n_gens = 100
pop_size = 100



# starting params from gosh et al
tm = 0.5 #s
AIB_v0 = AIA_v0 = AIY_v0 = AWC_v0 = 0
AWC_gain = 2
AWC_f_a = 4 #1/s
AWC_f_b = 15 #1/s
AWC_s_gamma = 2 #1/s
speed = 0.11 #mm/s

w_2 = w_3 = w_4 = w_5  = -2 # -ve weights
w_1 = w_6 = w_7 = 2 # +ve weights
w_8 = 0.5
w_9 = -0.5
dataset = sex_odour
print(len(dataset))
n_worms = len(dataset)
simulator = WormSimulator(dataset = dataset, dt = 0.005)




#sol = solve_ivp(xdot, t_span, y0, t_eval = np.arange(t_span[-1]), args = (p,)).y





opt = 'S'
#sol = forward_euler(y0, parameters, dt, t_span[-1])


#evolve_constraints()

#print(score_worm(sol))


worm_trapped = False
conc_interval = None
fit_w8_w9 = False

params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
          speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, worm_trapped, conc_interval]

path = '/Users/neythen/Desktop/Projects/worm_neural_networks/results/worm_simulation_results_NT/230115_sexual/fitting_output/'

'''
max:  -2.2976276924854786 [  1.59345591  -2.45687129  -3.98536621   1.          -5.47492077
  -7.92269622 -10.59328052  10.7723366   -1.        ]
'''

if opt == 'E': # evolve
    evolve_constraints(simulator, n_gens, pop_size)
elif opt == 'P':  # plot
    population = np.load(path + 'population.npy')
    fitnesses = np.load(path + 'fitnesses.npy')

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

    plt.show()

elif opt == 'T': # test
    n_worms = 1000

    population = np.load(path + 'gen226/population.npy')
    fitnesses = np.load(path + 'gen226/fitnesses.npy')
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
        path + 'weights_population.npy')

    fitnesses = np.load(path + 'final_fitnesses.npy')

    order = np.argsort(fitnesses)[::-1]
    population = population[order]

    p = population[0]


    # positive weights
    params[10] = p[0]
    params[15] = p[1]
    params[16] = p[2]
    params[17] = p[3]

    # negative weights
    params[11] = p[4]
    params[12] = p[5]
    params[13] = p[6]
    params[14] = p[7]
    params[18] = p[8]

    t = time.time()

    sol = simulator.forward_euler(simulator.y0, params)

    print(time.time()- t)

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
        path + 'weights_population.npy')
    ncols = 10
    fig, axs = plt.subplots(nrows=10, ncols=ncols, figsize=(15, 7.5))
    for i in range(len(population)):
        p = population[i]

        # positive weights
        params[10] = p[0]
        params[15] = p[1]
        params[16] = p[2]
        params[17] = p[3]

        # negative weights
        params[11] = p[4]
        params[12] = p[5]
        params[13] = p[6]
        params[14] = p[7]
        params[18] = p[8]

        simulator.t_span[-1] = max_t
        solution = simulator.forward_euler(simulator.y0, params)

        # plot neuron voltages
        ax = axs[i // ncols, i % ncols]
        ax.set_ylim(bottom=-1., top=1.)
        ax.plot(np.arange(0, max_t, simulator.dt), solution[0, 1:-1], label='AWC')
        ax.plot(np.arange(0, max_t, simulator.dt), solution[3, 1:-1], label='AIB')
        ax.plot(np.arange(0, max_t, simulator.dt), solution[5, 1:-1], label='AIY')
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron voltages')
        # simulator.plot_conc()
    plt.show()