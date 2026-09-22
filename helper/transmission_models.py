import numpy as np

def general_age_seir(state, t, beta, C, phi, sig, gam):
    """
    n-age class SIR ODEs
    (state) 1D array of compartments; must have length (5n)
        four blocks [S1,S2,...,Sn, E1,E2,...,En, I1,I2,...,In, R1,R2,...,Rn, Y1,Y2,...,Yn]
        Si: susceptible
        Ei: exposed
        Ii: infected
        Ri: removed
        Yi: cumulative infected
    (beta) transmission parameter function (of t); must be a function that takes a single input
    (C) age contact matrix; must have shape (nxn)
        Cij: contact rate between individuals of class i with individuals of class j
    (phi) contact rate scaling parameters; must have shape (nx1)
        phi_i: contact rate scaling parameter for individuals of class i
    (sig) exposed class exit rates; must have shape (nx1)
        sig_i: exposed class exit rate for individuals of class i
    (gam) recovery rates; must have shape (nx1)
        gam_i: recovery rate for individuals of class i
    """

    # extract states
    n = int(len(state) / 5) # number of age classes
    state_vec = state.reshape(5*n, 1) # reshape to column vector
    S = state_vec[0*n:1*n, :]
    E = state_vec[1*n:2*n, :]
    I = state_vec[2*n:3*n, :]
    R = state_vec[3*n:4*n, :]
    Y = state_vec[4*n:5*n, :]

    # transmission and recovery terms
    N = S + E + I + R # number of individuals in each age class
    lam = beta(t)*(C@(I/N))*phi # force of infection for each age class
    dS = -lam*S
    dE = lam*S - sig*E
    dI = sig*E - gam*I
    dR = gam*I
    dY = sig*E

    return np.array([dS, dE, dI, dR, dY]).flatten()

def pitzer_sirsirsirs(St, t, w0, w1, w2, g1, g2, s1, s2, p2, pA, beta, seasonality, phi):
    
    T = 12*50

    # starting proportion of people in each age class
    agep = np.concatenate(((1/960)*np.ones(12), (1/80)*np.ones(4), np.array([1/16, 1/8, 1/4, 1/4, 1/4])))
    al = len(agep)
    u = np.concatenate((1*np.ones(12), (1/12)*np.ones(4), np.array([1/(12*5), 1/120, 1/240, 1/240, 1/126])))
    # age-specific susceptibility
    c = np.concatenate((1.555*np.ones(12), np.array([2.312,1.738]), np.ones(7)))

    Byr = 0.017  # Birth rate per year
    Bca = np.array([0.0206, 0.0200, 0.0194, 0.0187, 0.0180, 0.0174, 0.0168, 0.0162, 0.0158, 0.0155, 0.0157, 0.0153, 0.0151, 0.0152, 0.0152, 0.0152, 0.0154])
    Bus = np.array([0.0167, 0.0162, 0.0158, 0.0154, 0.0150, 0.0146, 0.0144, 0.0142, 0.0143, 0.0142, 0.0144, 0.0141, 0.0139, 0.0141, 0.0140, 0.0140, 0.0142])
    B = np.zeros((T,al))
    B[0:240,0] = Byr
    B[240:300,0] = Bus[0]
    B[300:(300+15*12),0] = np.repeat(Bca[1:16],12)
    B[(300+15*12):(300+15*12+120),0] = Bca[16]

    # vaccination
    v = np.zeros((T,al))
    v[:,0] = np.concatenate((np.zeros(480), np.ones(120)))*0.96*(8/10)

    t_ = np.min([int(t), T-1])
    dSt = np.zeros(len(St))
    foi = beta*(1+seasonality*np.cos(2*np.pi*(t/12-phi)))*np.sum(St[2*al:3*al]+p2*St[5*al:6*al]+pA*St[8*al:9*al])*c
    # M
    dSt[0:al] = (1-v[int(t_),:])*np.log(1+B[int(t_)-1,:])*np.sum(St)/12 - w0*St[0:al] - u*St[0:al] + np.concatenate((np.zeros(1), u[:-1]*St[0:al-1]))
    # S0
    dSt[al:2*al] = -foi*St[al:2*al] + w0*St[0:al] - u*St[al:2*al] + np.concatenate((np.zeros(1), u[:-1]*St[al:2*al-1]))
    # I1
    dSt[2*al:3*al] = foi*St[al:2*al] - g1*St[2*al:3*al] - u*St[2*al:3*al] + np.concatenate((np.zeros(1), u[:-1]*St[2*al:3*al-1]))
    # R1
    dSt[3*al:4*al] = g1*St[2*al:3*al] - w1*St[3*al:4*al] - u*St[3*al:4*al] + np.concatenate((np.zeros(1), u[:-1]*St[3*al:4*al-1]))
    # S1
    dSt[4*al:5*al] = v[int(t_),:]*np.log(1+B[int(t_)-1,:])*np.sum(St)/12 + w1*St[3*al:4*al] - s1*foi*St[4*al:5*al] - u*St[4*al:5*al] + np.concatenate((np.zeros(1), u[:-1]*St[4*al:5*al-1]))
    # I2
    dSt[5*al:6*al] = s1*foi*St[4*al:5*al] - g2*St[5*al:6*al] - u*St[5*al:6*al] + np.concatenate((np.zeros(1), u[:-1]*St[5*al:6*al-1]))
    # R2
    dSt[6*al:7*al] = g2*St[5*al:6*al] - w1*St[6*al:7*al] - u*St[6*al:7*al] + np.concatenate((np.zeros(1), u[:-1]*St[6*al:7*al-1]))
    # S2
    dSt[7*al:8*al] = w1*St[6*al:7*al] + w2*St[9*al:10*al] - s2*foi*St[7*al:8*al] - u*St[7*al:8*al] + np.concatenate((np.zeros(1), u[:-1]*St[7*al:8*al-1]))
    # IA
    dSt[8*al:9*al] = s2*foi*St[7*al:8*al] - g2*St[8*al:9*al] - u*St[8*al:9*al] + np.concatenate((np.zeros(1), u[:-1]*St[8*al:9*al-1]))
    # RA
    dSt[9*al:10*al] = g2*St[8*al:9*al] - w2*St[9*al:10*al] - u*St[9*al:10*al] + np.concatenate((np.zeros(1), u[:-1]*St[9*al:10*al-1]))
    
    return(dSt)