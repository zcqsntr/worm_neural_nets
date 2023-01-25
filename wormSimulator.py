import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat

class WormSimulator():
    def __init__(self, dataset, dt, t_span=[0,1200], y0=[0, 0, 0, 0, 0, 0, 0, 0]):
        self.params =[4, 15, 2, 0.5, 0, 2, 0, 0, 0, 0.11, False, None] # [AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0,speed, worm_trapped, conc_interval]
        self.dt = dt
        self.t_span = t_span  # s
        self.y0 = y0
        self.theta = np.random.random() * 2 * np.pi
        self.dataset = dataset
        self.ms = np.mean(list(map(sum, self.dataset)))
        self.ss = np.std(list(map(sum, self.dataset)))
        self.sks = stats.skew(list(map(sum, self.dataset)))

        self.mr = np.mean(list(map(lambda x: max(x) - min(x), self.dataset)))
        self.sr = np.std(list(map(lambda x: max(x) - min(x), self.dataset)))
        self.skr = stats.skew(list(map(lambda x: max(x) - min(x), self.dataset)))
        self.plate_r = 3
        self.origin = np.array([4.5, 0])
        self.domain =[-3,3]
        self.sample_time = 0.1
        self.last_sample = 0

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def concentration_func(self, x,y,t, t_interval = None):

        origin = np.array([4.5, 0.])

        delx = origin[0] - x
        dely = origin[1] - y

        dist = np.sqrt(delx**2 + dely**2)

        std = 4
        if t_interval is None or t_interval[0] <= t <= t_interval[1]:

            if t_interval is None:
                return self.gaussian(dist, mu=0, sig=std)*100
            else:
                return self.gaussian(dist, mu=0, sig=std)*5
        else:
            return 0

    def xdot(self, t, X, weights, p):

        plate_r = 3

        w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9 = weights

        AWC_v, AWC_f, AWC_s, AIB_v, AIA_v, AIY_v, x, y = X

        AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0, speed, worm_trapped, conc_interval = p

        conc = self.concentration_func(x, y, t, conc_interval)

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

        if worm_trapped:
            dy = dx = 0
        else:
            turn = False
            if self.last_sample >= self.sample_time:
                turn = (np.random.random() * 2 - 1) < np.tanh(w_8*AIB_v + w_9*AIY_v) # dt = sample_time so just make this decision every time
                self.last_sample = 0
            else:
                self.last_sample += self.dt

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
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(25, 7.5))

        t = np.arange(len(solution[0, :]))*self.dt
        # plot sensory neuron components
        axs[0].plot(t, solution[1, :], label='AWC fast')
        axs[0].plot(t, solution[2, :], label='AWC slow')
        axs[0].legend()
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Sensory neuron voltage')

        #plot neuron voltages
        axs[1].plot(t,solution[0, :], label = 'AWC')
        axs[1].plot(t,solution[3, :], label = 'AIB')
        axs[1].plot(t,solution[4, :], label = 'AIA')
        axs[1].plot(t,solution[5, :], label = 'AIY')
        axs[1].legend()
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Neuron voltages')


        # plot worm position


        # plate outline
        circle = plt.Circle([0,0], self.plate_r, fill=False)
        axs[2].add_patch(circle)

        # scoring sectors
        pos = [-self.origin, self.origin]
        rad = [2.5, 3.5]

        for p in pos:
            for r in rad:
                circle = plt.Circle(p, r, fill=False, color='gray')
                axs[2].add_patch(circle)

        axs[2].vlines(0, self.domain[0], self.domain[1], color='grey')

        axs[2].plot(solution[6,:], solution[7,:])
        axs[2].scatter(solution[6,0], solution[7,0], label = 'start')
        axs[2].scatter(solution[6,-1], solution[7,-1], label = 'end')


        axs[2].set_xlim(self.domain[0], self.domain[1])
        axs[2].set_ylim(self.domain[0], self.domain[1])

        axs[2].legend(loc = 'lower left')

        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'worm.pdf'))


    def plot_conc(self):
        plt.figure()
        x = np.arange(self.domain[0], self.domain[1], 0.001)
        y = np.arange(self.domain[0], self.domain[1], 0.001)
        X, Y = np.meshgrid(x, y)

        img = self.concentration_func(X,Y,0)

        plt.imshow(img, extent=[self.domain[0], self.domain[1], self.domain[1], self.domain[0]])
        plt.colorbar()


    def score_worm(self, solution):
        trajectory = solution[6:8, :]
        x = trajectory[0, :]
        y = trajectory[1, :]
        delx = self.origin[0] - x
        dely = self.origin[1] - y

        origin_dist = np.sqrt(delx ** 2 + dely ** 2)

        mirror_origin = -self.origin
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

    def run_experiment(self, weights, n_worms):
        sectors = []

        for i in range(n_worms):

            self.theta = np.random.random() * 2 * np.pi
            self.last_sample = 0

            sol = self.forward_euler(self.y0, weights)

            sector = self.score_worm(sol)

            sectors.append(sector)


        return sectors

    def fitness_from_sectors(self, sectors):
        mean_score = np.mean(list(map(sum, sectors)))
        std_score = np.std(list(map(sum, sectors)))
        skew_score = stats.skew(list(map(sum, sectors)))
        mean_range = np.mean(list(map(lambda x: max(x) - min(x), sectors)))
        std_range = np.std(list(map(lambda x: max(x) - min(x), sectors)))
        skew_range = stats.skew(list(map(lambda x: max(x) - min(x), sectors)))

        fitness = - (abs(mean_score - self.ms) + abs(mean_range - self.mr) + abs(std_score - self.ss) + abs(std_range - self.sr) + abs(
            skew_score - self.sks) + abs(skew_range - self.skr))

        return fitness

    def get_fitness(self, weights, n_worms):

        params = self.params


        sectors = self.run_experiment(params, weights, n_worms)
        fitness = self.fitness_from_sectors(sectors)

        #print((abs(mean_score - ms), abs(mean_range - mr), abs(std_score - ss),abs(std_range - sr),abs(skew_score - sks),abs(skew_range - skr)))
        return fitness

    def get_fitnesses(self, population, n_worms):

        # values form the no cond no odor dat
        fitnesses = []


        for p in population:
            fitness = self.get_fitness(p, n_worms)
            '''
            print('score', abs(mean_score),  'score std',abs(std_score), 'range', abs(mean_range), 'range std', abs(std_range))
            print('score error', abs(mean_score-ms),'score std error', abs(std_score-ss),  'range error', abs(mean_range-mr), 'range std error', abs(std_range-sr))
            print('score error', ms,'score std error', ss,  'range error', mr, 'range std error',sr)
            print()
            '''
            fitnesses.append(fitness)

        print('done')
        return fitnesses

    def get_fitnesses_par(self, population, n_worms):
        n_cores = int(mp.cpu_count())

        with Pool(n_cores) as pool:
            fitnesses = pool.starmap(self.get_fitness, zip(population, repeat(n_worms)))

        return fitnesses

    def run_experiment_wrapper(self, weights, n_worms):

        sectors = self.run_experiment(weights, n_worms)
        return sectors

    def run_experiment_par(self, weights_population, n_worms):
        n_cores = int(mp.cpu_count())

        with Pool(n_cores) as pool:
            sectors = pool.starmap(self.run_experiment_wrapper, zip(weights_population, repeat(n_worms)))

        return sectors

    def forward_euler(self, y0, weights):
        y = y0
        all_ys = [y0]
        tmax = self.t_span[-1]
        for t in np.arange(0, tmax+self.dt, self.dt):
            y = y + np.array(self.xdot(t, y, weights, self.params))*self.dt
            all_ys.append(y)

        return np.array(all_ys).T