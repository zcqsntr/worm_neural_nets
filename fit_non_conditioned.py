
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import os
import time
from load_data import load_data
from multiprocessing import Pool
import multiprocessing as mp
from scipy import stats
import matplotlib.cm as cm
import matplotlib.animation as animation





class WormSimulator():
    def __init__(self, dataset, dt, t_span=[0,1200], y0=[0, 0, 0, 0, 0, 0, 0, 0]):
        self.params =[4, 15, 2, 0.5, 0, 2, 0, 0, 0, 0.11, 2, -2, -2, -2, -2, 2, 2, 0.5, -0.5] # [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]
        self.dt = dt
        self.t_span = t_span  # s
        self.y0 = y0
        self.theta = np.random.random() * 2 * np.pi
        self.dataset = dataset

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def concentration_func(self, x,y,t):

        origin = np.array([4.5, 0.])

        delx = origin[0] - x
        dely = origin[1] - y

        dist = np.sqrt(delx**2 + dely**2)

        std = 4

        return self.gaussian(dist, mu=0, sig=std)*100

    def xdot(self, t, X, p):

        plate_r = 3

        AWC_v, AWC_f, AWC_s, AIB_v, AIA_v, AIY_v, x, y = X

        AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0, speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9 = p

        conc = self.concentration_func(x, y, t)

        dAWC_f = AWC_f_a*conc - AWC_f_b*AWC_f
        dAWC_s = AWC_s_gamma*(AWC_f - AWC_s)
        AWC_i = AWC_f-AWC_s
        dAWC_v = 1/tm *(-AWC_v + AWC_v0 + np.tanh(-AWC_gain*AWC_i)) # -ve in tanh because downstep in conc activates AWC

        AIB_i = w_1*AWC_v + w_4*AIA_v
        dAIB_v = 1/tm *(-AIB_v + AIB_v0 + np.tanh(AIB_i)) # removed gains as redundant with the weights


        AIA_i = w_2 * AWC_v + w_5*AIB_v + w_6*AIY_v
        dAIA_v = 1 / tm * (-AIA_v + AIA_v0 + np.tanh(AIA_i))

        AIY_i = w_3 * AWC_v + w_7 * AIA_v
        dAIY_v = 1 / tm * (-AIY_v + AIY_v0 + np.tanh(AIY_i))

        #go_forward = (np.random.random()*2 - 1) < AIY_v

        turn = (np.random.random() * 2 - 1) < np.tanh(w_8*AIB_v + w_9*AIY_v) # dt = sample_time so just make this decision every time

        if turn:
            self.theta = np.random.random() * 2 * np.pi

        dx = speed * np.cos(self.theta)
        dy = speed * np.sin(self.theta)

        # stop worms going off the plate by choosing another random direction, this stops them getting stuck on the edge
        x, y = X[6], X[7]
        while abs((x+dx)**2 + (y+dy)**2)**0.5 > plate_r:
            self.theta = np.random.random() * 2 * np.pi
            dx = speed * np.cos(self.theta)
            dy = speed * np.sin(self.theta)



        return dAWC_v, dAWC_f, dAWC_s, dAIB_v, dAIA_v, dAIY_v, dx, dy

    def plot_sol(self, solution, save_path = None):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 7.5))

        # plot sensory neuron components
        axs[0].plot(np.arange(self.t_span[0], self.t_span[1] + 2*self.dt, self.dt), solution[1, :], label='AWC fast')
        axs[0].plot(np.arange(self.t_span[0], self.t_span[1] + 2*self.dt, self.dt), solution[2, :], label='AWC slow')
        axs[0].legend()
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Sensory neuron voltage')

        #plot neuron voltages
        axs[1].plot(np.arange(self.t_span[0], self.t_span[1] + 2*self.dt, self.dt),solution[0, :], label = 'AWC')
        axs[1].plot(np.arange(self.t_span[0], self.t_span[1] + 2*self.dt, self.dt),solution[3, :], label = 'AIB')
        axs[1].plot(np.arange(self.t_span[0], self.t_span[1] + 2*self.dt, self.dt),solution[4, :], label = 'AIA')
        axs[1].plot(np.arange(self.t_span[0], self.t_span[1] + 2*self.dt, self.dt),solution[5, :], label = 'AIY')
        axs[1].legend()
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Neuron voltages')


        # plot worm position


        # plate outline
        circle = plt.Circle([0,0], plate_r, fill=False)
        axs[2].add_patch(circle)

        # scoring sectors
        pos = [-origin, origin]
        rad = [2.5, 3.5]

        for p in pos:
            for r in rad:
                circle = plt.Circle(p, r, fill=False, color='gray')
                axs[2].add_patch(circle)

        axs[2].vlines(0, domain[0], domain[1], color='grey')

        axs[2].plot(solution[6,:], solution[7,:])
        axs[2].scatter(solution[6,0], solution[7,0], label = 'start')
        axs[2].scatter(solution[6,-1], solution[7,-1], label = 'end')


        axs[2].set_xlim(domain[0], domain[1])
        axs[2].set_ylim(domain[0], domain[1])

        axs[2].legend(loc = 'lower left')

        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'worm.pdf'))


    def plot_conc(self, domain):
        plt.figure()
        x = np.arange(domain[0], domain[1], 0.001)
        y = np.arange(domain[0], domain[1], 0.001)
        X, Y = np.meshgrid(x, y)

        img = self.concentration_func(X,Y,0)

        plt.imshow(img, extent=[domain[0], domain[1], domain[1], domain[0]])
        plt.colorbar()


    def score_worm(self, solution):
        trajectory = solution[6:8, :]
        x = trajectory[0, :]
        y = trajectory[1, :]
        delx = origin[0] - x
        dely = origin[1] - y

        origin_dist = np.sqrt(delx ** 2 + dely ** 2)

        mirror_origin = -origin
        delx = mirror_origin[0] - x
        dely = mirror_origin[1] - y

        mirror_origin_dist = np.sqrt(delx ** 2 + dely ** 2)

        sectors = []

        if np.any(mirror_origin_dist < 2.5):
            sectors.append(-3)
        if np.any(mirror_origin_dist < 3.5):
            sectors.append(-2)

        if np.any(origin_dist < 2.5):
            sectors.append(3)

        if np.any(origin_dist < 3.5):
            sectors.append(2)

        if np.any(x < 0):
            sectors.append(-1)
        if np.any(x > 0):
            sectors.append(1)

        return sectors


    def run_experiment(self, params, n_worms):



        sectors = []



        for i in range(n_worms):

            theta = np.random.random() * 2 * np.pi
            last_sample = 0

            sol = self.forward_euler(self.y0, params, self.dt, self.t_span[-1])

            sector = self.score_worm(sol)

            sectors.append(sector)


        return sectors

    def get_fitness(self, weights):

        ms = np.mean(list(map(sum, self.dataset)))
        ss = np.std(list(map(sum, self.dataset)))
        sks = stats.skew(list(map(sum, self.dataset)))

        mr = np.mean(list(map(lambda x: max(x) - min(x), self.dataset)))
        sr = np.std(list(map(lambda x: max(x) - min(x), self.dataset)))
        skr = stats.skew(list(map(lambda x: max(x) - min(x), self.dataset)))


        params = self.params

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

        sectors = self.run_experiment(params, n_worms)



        mean_score = np.mean(list(map(sum, sectors)))
        std_score = np.std(list(map(sum, sectors)))
        skew_score = stats.skew(list(map(sum, sectors)))
        mean_range = np.mean(list(map(lambda x: max(x) - min(x), sectors)))
        std_range = np.std(list(map(lambda x: max(x) - min(x), sectors)))
        skew_range = stats.skew(list(map(lambda x: max(x) - min(x), sectors)))


        fitness = - (abs(mean_score - ms) + abs(mean_range - mr) + abs(std_score - ss) + abs(std_range - sr) + abs(skew_score - sks) + abs(skew_range - skr))

        return fitness

    def get_fitnesses(self, population):

        # values form the no cond no odor dat
        fitnesses = []


        for p in population:
            fitness = self.get_fitness(p)
            '''
            print('score', abs(mean_score),  'score std',abs(std_score), 'range', abs(mean_range), 'range std', abs(std_range))
            print('score error', abs(mean_score-ms),'score std error', abs(std_score-ss),  'range error', abs(mean_range-mr), 'range std error', abs(std_range-sr))
            print('score error', ms,'score std error', ss,  'range error', mr, 'range std error',sr)
            print()
            '''
            fitnesses.append(fitness)


        return fitnesses


    def get_fitnesses_par(self, population):
        n_cores = int(mp.cpu_count())

        with Pool(n_cores) as pool:
            fitnesses = pool.map(self.get_fitness, population)

        return fitnesses

    def run_experiment_wrapper(self, p):
        params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
                  speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]

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

        sectors = self.run_experiment(params, n_worms)
        return sectors

    def run_experiment_par(self, population):
        n_cores = int(mp.cpu_count())

        with Pool(n_cores) as pool:
            sectors = pool.map(self.run_experiment_wrapper, population)

        return sectors


    def param_scan(self, start, stop, step, save_path = './working_dir/param_scan', plot=True):
        os.makedirs(save_path, exist_ok = True)
        population = np.arange(start, stop, step).reshape(-1, 1)

        fitnesses, all_sectors  = self.get_fitnesses(population)



        if plot:
            plt.plot(population, fitnesses)


        np.save(save_path + '/params.npy',population)
        np.save(save_path + '/fitnesses.npy',fitnesses)
        np.save(save_path + '/sectors.npy', all_sectors)

        if plot:
            plt.show()


    def forward_euler(self, y0, params, dt, tmax):
        AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0, speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9 = params


        y = y0
        all_ys = [y0]

        for t in np.arange(0, tmax+dt, dt):
            y = y + np.array(self.xdot(t, y, params))*dt
            all_ys.append(y)

        return np.array(all_ys).T


def evolve(simulator, n_gens, pop_size, save_path = './working_dir/evolution'):



    population = np.random.random(size = (pop_size, 9))*20 - 10



    for i in range(n_gens):
        save_p = os.path.join(save_path, 'gen' + str(i))
        os.makedirs(save_p, exist_ok=True)
        fitnesses = simulator.get_fitnesses(population)
        np.save(save_p + '/population.npy', population)
        np.save(save_p + '/fitnesses.npy', fitnesses)



        order = np.argsort(fitnesses)[::-1]

        fitnesses = np.array(fitnesses)[order]

        population = population[order]

        population[int(pop_size*0.4): int(pop_size*0.8)] += np.random.random(size = (int(pop_size*0.8)- int(pop_size*0.4), 9))*2 - 1.


        population[int(pop_size*0.8):] = np.random.random(size = (pop_size-int(pop_size*0.8), 9))*20-10


        print('max: ', np.max(fitnesses), population[0])
        print('mean: ', np.mean(fitnesses))


no_cond_no_odour, no_cond_odour, aversive_odour, sex_odour = load_data('./data/behaviourdatabysector_NT.csv')

n_gens = 1000
pop_size = 100
n_worms = 70 # number of worms in each experiment

origin = np.array([4.5, 0.])
# starting params from gosh et al
tm = 0.5 #s
AIB_v0 = AIA_v0 = AIY_v0 = AWC_v0 = 0
AWC_gain = 2
AWC_f_a = 4 #1/s
AWC_f_b = 15 #1/s
AWC_s_gamma = 2 #1/s
speed = 0.11 #mm/s

domain = [-3,3]
plate_r = 3
w_2 = w_3 = w_4 = w_5  = -2 # -ve weights
w_1 = w_6 = w_7 = 2 # +ve weights
w_8 = 0.5
w_9 = -0.5

simulator = WormSimulator(dataset = no_cond_no_odour, dt = 0.1)




#sol = solve_ivp(xdot, t_span, y0, t_eval = np.arange(t_span[-1]), args = (p,)).y




opt = 'P'
#sol = forward_euler(y0, parameters, dt, t_span[-1])


#evolve_constraints()

#print(score_worm(sol))




if opt == 'E':
    evolve(simulator, n_gens, pop_size)
elif opt == 'P':
    population = np.load('/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/281122_fit_constrained/population.npy')
    fitnesses = np.load('/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/281122_fit_constrained/fitnesses.npy')

    order = np.argsort(fitnesses)[::-1]
    population = population[order]

    params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
              speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]

    all_sectors = []



    all_sectors = simulator.run_experiment_par(population[0:25])

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

        ax.violinplot(list(map(sum, no_cond_odour)))
        ax.violinplot(list(map(sum, sectors)))



    fig.suptitle('Violin plots of top 25 members of the evolutionary population')
    plt.savefig('violin_plots.png', dpi=300)






    print('score', np.mean(list(map(sum, no_cond_odour))), 'score std', np.std(list(map(sum, no_cond_odour))), 'range',
          np.mean(list(map(lambda x: max(x) - min(x), no_cond_odour))), 'range std',
          np.std(list(map(lambda x: max(x) - min(x), no_cond_odour))))


    print('score', np.mean(list(map(sum, sectors))), 'score std', np.std(list(map(sum, sectors))), 'range',
          np.mean(list(map(lambda x: max(x) - min(x), sectors))), 'range std',
          np.std(list(map(lambda x: max(x) - min(x), sectors))))

    plt.show()

elif opt == 'T':
    n_worms = 1000
    path = '/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/281122_fit_constrained/gen501/'
    population = np.load(path + 'population.npy')
    fitnesses = np.load(path + 'fitnesses.npy')
    order = np.argsort(fitnesses)[::-1]
    print(order)

    t = time.time()
    fitnesses = simulator.get_fitnesses_par(population)
    print(time.time() - t)
    order = np.argsort(fitnesses)[::-1]
    print(order)

    np.save('/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/281122_fit_constrained/' + 'population.npy', population)
    np.save('/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/281122_fit_constrained/' + 'fitnesses.npy', fitnesses)

elif opt == 'S':
    population = np.load(
        '/home/neythen/Desktop/Projects/worm_neural_nets/results/fitting_unconditioned/281122_fit_constrained/population.npy')
    p = population[0]

    params = [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,
              speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]

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

    sol = simulator.forward_euler(simulator.y0, params, simulator.dt, simulator.t_span[-1])

    print(simulator.score_worm(sol))

    simulator.plot_sol(sol)
    simulator.plot_conc(domain)
    plt.show()