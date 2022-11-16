import numpy as np


theta = np.random.random() * 2 * np.pi
last_sample = 0
go_forward = True

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def concentration_func(x,y,t):

    origin = np.array([4.5, 0.])

    delx = origin[0] - x
    dely = origin[1] - y

    dist = np.sqrt(delx**2 + dely**2)

    std = 4

    return gaussian(dist, mu=0, sig=std)*100


def xdot(t, X, p):
    global theta
    global last_sample
    global go_forward


    plate_r = 3

    AWC_v, AWC_f, AWC_s, AIB_v, AIA_v, AIY_v, x, y = X

    AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0, speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, sampling_time = p

    conc = concentration_func(x, y, t)

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

    turn = False

    if t - last_sample >= sampling_time:
        last_sample = t
        #turn = (np.random.random() * 2 - 1) < AIB_v
        #go_forward = (np.random.random() * 2 - 1) < AIY_v

        turn = (np.random.random() * 4 - 2) < (AIB_v - AIY_v)

    if turn:
        theta = np.random.random() * 2 * np.pi

    dx = speed * np.cos(theta)
    dy = speed * np.sin(theta)

    # stop worms going off the plate
    x, y = X[6], X[7]
    if abs((x+dx)**2 + (y+dy)**2)**0.5 > plate_r:
        dx = dy = 0

    return dAWC_v, dAWC_f, dAWC_s, dAIB_v, dAIA_v, dAIY_v, dx, dy



def xdot_RIA(t, X, p):
    global theta
    AWC_v, AWC_f, AWC_s, AIB_v, AIA_v, AIY_v, RIA_v, x, y = X

    AWC_f_a, AWC_f_b, AWC_s_gamma, tm, AWC_v0, AWC_gain, AIB_v0, AIA_v0, AIY_v0, RIA_v0, speed, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8 = p

    conc = concentration_func(x, y, t)

    dAWC_f = AWC_f_a*conc - AWC_f_b*AWC_f
    dAWC_s = AWC_s_gamma*(AWC_f - AWC_s)
    AWC_i = AWC_f-AWC_s
    dAWC_v = 1/tm *(-AWC_v + AWC_v0 + np.tanh(-AWC_gain*AWC_i)) # -ve in tanh because downstep in conc activates AWC

    AIB_i = w_1*AWC_v + w_4*AIA_v
    dAIB_v = 1/tm *(-AIB_v + AIB_v0 + np.tanh(AIB_i)) # removed gains as redundant with the weights
    #dAIB_v = 0 # conditioned worm


    AIA_i = w_2 * AWC_v + w_5*AIB_v + w_6*AIY_v
    dAIA_v = 1 / tm * (-AIA_v + AIA_v0 + np.tanh(AIA_i))

    AIY_i = w_3 * AWC_v + w_7 * AIA_v
    dAIY_v = 1 / tm * (-AIY_v + AIY_v0 + np.tanh(AIY_i))

    RIA_i = w_8 * AIY_v
    dRIA_v = 1 / tm * (-RIA_v + RIA_v0 + np.tanh(RIA_i))


    turn = (np.random.random()*2 - 1) < AIB_v or ((np.random.random()*2 - 1) < RIA_v)
    go_forward = (np.random.random()*2 - 1) < AIY_v

    if turn:
        theta = np.random.random() * 2 * np.pi

    if go_forward:
        dx = speed * np.cos(theta)
        dy = speed * np.sin(theta)
    else:
        dx = dy = 0

    return dAWC_v, dAWC_f, dAWC_s, dAIB_v, dAIA_v, dAIY_v, dRIA_v, dx, dy





