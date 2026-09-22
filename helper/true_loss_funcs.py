import numpy as np
from transmission_models import general_age_seir, pitzer_sirsirsirs
from helper import collapse_contact_matrix, find_group_dist
from scipy.integrate import odeint
from scipy.stats import poisson

def generate_neg_log_likelihood_4_age(phis, rng):

    phis = np.array(phis)

    # read Prem et al. contact matrix
    prem_US = np.loadtxt("contact_mx/prem_US_all_locations.csv", delimiter = ",")
    age_dist = find_group_dist(prem_US)

    # collapse Prem et al. contact matrix to four groups
    group_encoding = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]])
    C = collapse_contact_matrix(prem_US, group_encoding, age_dist)
    collapsed_age_dist = age_dist.reshape(4, 4).sum(axis = 1)

    n = 4
    beta = lambda t: 0.1
    phi = phis.reshape(n,1)
    sig = np.array([1/5]*n).reshape(n,1)
    gam = np.array([1/7]*n).reshape(n,1)
    args = (beta, C, phi, sig, gam)

    S0 =  collapsed_age_dist * 1e+4
    E0 = np.array([0.]*n)
    I0 = np.array([0.]*n); I0[0] = 1.
    R0 = np.array([0.]*n)
    Y0 = np.array([0.]*n)
    y0 = np.array([S0, E0, I0, R0, Y0]).flatten()

    t = np.arange(0, 50, 1)

    def solve_ode(args):
        return odeint(general_age_seir, y0, t, args = args)

    sol = solve_ode(args)

    # compute incidence
    cum_cases = sol[:, (4*n):(5*n)]
    incs = np.diff(cum_cases, axis = 0)

    # generate observed data
    obs_cases = rng.poisson(incs)

    def log_likelihood(parms):
        parms = np.array(parms)

        # extract parameters
        beta = lambda t: 0.1
        phi = parms.reshape(n,1)
        sig = np.array([1/5]*n).reshape(n,1)
        gam = np.array([1/7]*n).reshape(n,1)

        # solve ODE
        args = (beta, C, phi, sig, gam)
        sol = solve_ode(args)

        # compute incidence
        cum_cases = sol[:, (4*n):(5*n)]
        incs = np.diff(cum_cases, axis = 0)
        np.clip(incs, a_min = 1e-10, a_max = None, out = incs)

        # compute log likelihood
        return np.sum(poisson.logpmf(obs_cases, incs))

    def loss_neg_log_likelihood(parms):
        return -1 * log_likelihood(parms)
    
    return loss_neg_log_likelihood

def generate_neg_log_likelihood_8_age(phis, rng):

    phis = np.array(phis)

    # read Prem et al. contact matrix
    prem_US = np.loadtxt("contact_mx/prem_US_all_locations.csv", delimiter = ",")
    age_dist = find_group_dist(prem_US)

    # collapse Prem et al. contact matrix to eight groups
    group_encoding = np.array([[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                               [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
                               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]])
    C = collapse_contact_matrix(prem_US, group_encoding, age_dist)
    collapsed_age_dist = age_dist.reshape(8, 2).sum(axis = 1)

    n = 8
    beta = lambda t: 0.1
    phi = phis.reshape(n,1)
    sig = np.array([1/5]*n).reshape(n,1)
    gam = np.array([1/7]*n).reshape(n,1)
    args = (beta, C, phi, sig, gam)

    S0 =  collapsed_age_dist * 1e+4
    E0 = np.array([0.]*n)
    I0 = np.array([0.]*n); I0[0] = 1.
    R0 = np.array([0.]*n)
    Y0 = np.array([0.]*n)
    y0 = np.array([S0, E0, I0, R0, Y0]).flatten()

    t = np.arange(0, 50, 1)

    def solve_ode(args):
        return odeint(general_age_seir, y0, t, args = args)

    sol = solve_ode(args)

    # compute incidence
    cum_cases = sol[:, (4*n):(5*n)]
    incs = np.diff(cum_cases, axis = 0)

    # generate observed data
    obs_cases = rng.poisson(incs)

    def log_likelihood(parms):
        parms = np.array(parms)

        # extract parameters
        beta = lambda t: 0.1
        phi = parms.reshape(n,1)
        sig = np.array([1/5]*n).reshape(n,1)
        gam = np.array([1/7]*n).reshape(n,1)

        # solve ODE
        args = (beta, C, phi, sig, gam)
        sol = solve_ode(args)

        # compute incidence
        cum_cases = sol[:, (4*n):(5*n)]
        incs = np.diff(cum_cases, axis = 0)
        np.clip(incs, a_min = 1e-10, a_max = None, out = incs)

        # compute log likelihood
        return np.sum(poisson.logpmf(obs_cases, incs))

    def loss_neg_log_likelihood(parms):
        return -1 * log_likelihood(parms)
    
    return loss_neg_log_likelihood

def generate_neg_log_likelihood_16_age(phis, rng):

    phis = np.array(phis)

    # read Prem et al. contact matrix
    prem_US = np.loadtxt("contact_mx/prem_US_all_locations.csv", delimiter = ",")
    age_dist = find_group_dist(prem_US)
    C = prem_US

    n = 16
    beta = lambda t: 0.1
    phi = phis.reshape(n,1)
    sig = np.array([1/5]*n).reshape(n,1)
    gam = np.array([1/7]*n).reshape(n,1)
    args = (beta, C, phi, sig, gam)

    S0 = age_dist * 1e+4
    E0 = np.array([0.]*n)
    I0 = np.array([0.]*n); I0[0] = 1.
    R0 = np.array([0.]*n)
    Y0 = np.array([0.]*n)
    y0 = np.array([S0, E0, I0, R0, Y0]).flatten()

    t = np.arange(0, 50, 1)

    def solve_ode(args):
        return odeint(general_age_seir, y0, t, args = args)

    sol = solve_ode(args)

    # compute incidence
    cum_cases = sol[:, (4*n):(5*n)]
    incs = np.diff(cum_cases, axis = 0)

    # generate observed data
    obs_cases = rng.poisson(incs)

    def log_likelihood(parms):
        parms = np.array(parms)

        # extract parameters
        beta = lambda t: 0.1
        phi = parms.reshape(n,1)
        sig = np.array([1/5]*n).reshape(n,1)
        gam = np.array([1/7]*n).reshape(n,1)

        # solve ODE
        args = (beta, C, phi, sig, gam)
        sol = solve_ode(args)

        # compute incidence
        cum_cases = sol[:, (4*n):(5*n)]
        incs = np.diff(cum_cases, axis = 0)
        np.clip(incs, a_min = 1e-10, a_max = None, out = incs)

        # compute log likelihood
        return np.sum(poisson.logpmf(obs_cases, incs))

    def loss_neg_log_likelihood(parms):
        return -1 * log_likelihood(parms)
    
    return loss_neg_log_likelihood

def generate_neg_log_likelihood_pitzer_4(rng):

    T = 12*50

    # starting proportion of people in each age class
    agep = np.concatenate(((1/960)*np.ones(12), (1/80)*np.ones(4), np.array([1/16, 1/8, 1/4, 1/4, 1/4])))
    al = len(agep)
    # age-specific susceptibility
    c = np.concatenate((1.555*np.ones(12), np.array([2.312,1.738]), np.ones(7)))

    # initial population size
    N = 2.9e7

    # initial state
    St0 = np.concatenate((N/960*np.ones(1), np.zeros(al-1), N*agep - np.ones(al), np.ones(al), np.zeros(7*al)))

    # Parameters
    w0 = (1/3)
    w1 = (1/9) 
    w2 = (1/12)
    g1 = 4.3
    g2 = 8.6
    s1 = 0.62
    s2 = 0.35
    p2 = 0.5
    pA = 0.1
    d1 = 0.11
    d2 = 0.029

    ptrans = 23.25
    b = ptrans*g1
    beta = b/N # UNKNOWN
    seasonality = 0.055 # UNKNOWN
    phi = 0.636 # UNKNOWN
    h = 0.041 # UNKNOWN

    args = (w0, w1, w2, g1, g2, s1, s2, p2, pA, beta, seasonality, phi)
    result = odeint(pitzer_sirsirsirs, St0, np.arange(0,T), args = args)

    # compute true number of hospitalizations
    betas = beta*(1+seasonality*np.cos(2*np.pi*(np.arange(0,T)/12-phi)))
    betas = betas.reshape((600, 1))
    num_I = np.sum(result[:,2*al:3*al]+p2*result[:,5*al:6*al]+pA*result[:,8*al:9*al], axis=1).reshape((600, 1)) 
    foi = betas*num_I*c
    H = h*d1*foi*result[:,al:2*al] + h*d2*s1*foi*result[:,4*al:5*al]
    H = np.sum(H, axis=1)

    # simulate observed data
    obs_H = rng.poisson(H)

    def llik(parms):

        # extract parameters
        beta = parms[0] / 1e+5
        seasonality = parms[1]
        phi = parms[2]
        h = parms[3]

        # Parameters
        w0 = (1/3)
        w1 = (1/9)
        w2 = (1/12)
        g1 = 4.3
        g2 = 8.6
        s1 = 0.62
        s2 = 0.35
        p2 = 0.5
        pA = 0.1
        d1 = 0.11
        d2 = 0.029
        
        # solve ODE
        args = (w0, w1, w2, g1, g2, s1, s2, p2, pA, beta, seasonality, phi)
        result = odeint(pitzer_sirsirsirs, St0, np.arange(0,T), args = args)

        # extract number of hospitalizations
        betas = beta*(1+seasonality*np.cos(2*np.pi*(np.arange(0,T)/12-phi)))
        betas = betas.reshape((600, 1))
        num_I = np.sum(result[:,2*al:3*al]+p2*result[:,5*al:6*al]+pA*result[:,8*al:9*al], axis=1).reshape((600, 1)) 
        foi = betas*num_I*c
        H = h*d1*foi*result[:,al:2*al] + h*d2*s1*foi*result[:,4*al:5*al]
        H = np.sum(H, axis=1)
        np.clip(H, a_min = 1e-10, a_max = None, out = H)

        # compute log likelihood
        return np.sum(poisson.logpmf(obs_H[12*25-24:T-24], H[12*25-24:T-24]))

    def neg_llik(parms):
        val = llik(parms)
        if np.isnan(val) or np.isinf(val):
            return 1e20
        return -1 * val

    return neg_llik

# harder problem with six unknown parameters
def generate_neg_log_likelihood_pitzer_6(rng):

    T = 12*50

    # starting proportion of people in each age class
    agep = np.concatenate(((1/960)*np.ones(12), (1/80)*np.ones(4), np.array([1/16, 1/8, 1/4, 1/4, 1/4])))
    al = len(agep)
    # age-specific susceptibility
    c = np.concatenate((1.555*np.ones(12), np.array([2.312,1.738]), np.ones(7)))

    # initial population size
    N = 2.9e7

    # initial state
    St0 = np.concatenate((N/960*np.ones(1), np.zeros(al-1), N*agep - np.ones(al), np.ones(al), np.zeros(7*al)))

    # Parameters
    w0 = (1/3)
    w1 = (1/9) 
    w2 = (1/12)
    g1 = 4.3
    g2 = 8.6
    s1 = 0.62
    s2 = 0.35
    p2 = 0.5 # UNKNOWN
    pA = 0.1 # UNKNOWN
    d1 = 0.11
    d2 = 0.029

    ptrans = 23.25
    b = ptrans*g1
    beta = b/N # UNKNOWN
    seasonality = 0.055 # UNKNOWN
    phi = 0.636 # UNKNOWN
    h = 0.041 # UNKNOWN

    args = (w0, w1, w2, g1, g2, s1, s2, p2, pA, beta, seasonality, phi)
    result = odeint(pitzer_sirsirsirs, St0, np.arange(0,T), args = args)

    # compute true number of hospitalizations
    betas = beta*(1+seasonality*np.cos(2*np.pi*(np.arange(0,T)/12-phi)))
    betas = betas.reshape((600, 1))
    num_I = np.sum(result[:,2*al:3*al]+p2*result[:,5*al:6*al]+pA*result[:,8*al:9*al], axis=1).reshape((600, 1)) 
    foi = betas*num_I*c
    H = h*d1*foi*result[:,al:2*al] + h*d2*s1*foi*result[:,4*al:5*al]
    H = np.sum(H, axis=1)

    # simulate observed data
    obs_H = rng.poisson(H)

    def llik(parms):

        # extract parameters
        beta = parms[0] / 1e+5
        seasonality = parms[1]
        phi = parms[2]
        h = parms[3]

        # Parameters
        w0 = (1/3)
        w1 = (1/9)
        w2 = (1/12)
        g1 = 4.3
        g2 = 8.6
        s1 = 0.62
        s2 = 0.35
        p2 = parms[4]
        pA = parms[5]
        d1 = 0.11
        d2 = 0.029
        
        # solve ODE
        args = (w0, w1, w2, g1, g2, s1, s2, p2, pA, beta, seasonality, phi)
        result = odeint(pitzer_sirsirsirs, St0, np.arange(0,T), args = args)

        # extract number of hospitalizations
        betas = beta*(1+seasonality*np.cos(2*np.pi*(np.arange(0,T)/12-phi)))
        betas = betas.reshape((600, 1))
        num_I = np.sum(result[:,2*al:3*al]+p2*result[:,5*al:6*al]+pA*result[:,8*al:9*al], axis=1).reshape((600, 1)) 
        foi = betas*num_I*c
        H = h*d1*foi*result[:,al:2*al] + h*d2*s1*foi*result[:,4*al:5*al]
        H = np.sum(H, axis=1)
        np.clip(H, a_min = 1e-10, a_max = None, out = H)

        # compute log likelihood
        return np.sum(poisson.logpmf(obs_H[12*25-24:T-24], H[12*25-24:T-24]))

    def neg_llik(parms):
        val = llik(parms)
        if np.isnan(val) or np.isinf(val):
            return 1e20
        return -1 * val

    return neg_llik