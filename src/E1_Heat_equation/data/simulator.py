'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import os
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def propagate(u0, u, D, dt, dx2, dy2):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ( (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2 + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )
    u0 = u.copy()
    return u0, u


def propagate2(u0, u, D, dt, dx2):
    gamma = dt/dx2*D
    for i in range(1, 63):
        for j in range(1, 63):
            u[i, j] = gamma * (u0[i+1][j] + u0[i-1][j] + u0[i][j+1] + u0[i][j-1] - 4*u0[i][j]) + u0[i][j]
    u0 = u.copy()
    return u0, u


def diffusionSim(D, ncases, prob_dis, saveDir):
    nel = 64
    
    x = np.linspace(0, 1, nel+2)
    xx, yy = np.meshgrid(x, x)
    
    dx, dy = 1.0/(nel+1), 1.0/(nel+1)
    dx2, dy2 = dx*dx, dy*dy

    dt = 0.01

    print('-'*37)
    print('Simulation model start')

    for i in range(ncases):
        case_dir = saveDir + '/run{:d}'.format(i)
        mkdir(case_dir)
        
        if prob_dis == 'uniform':
            u0 = np.random.uniform(low=0.0, high=1.0, size=(nel+2, nel+2))
        elif prob_dis == 'gaussian1':
            b = np.random.normal(0.5, 0.1, 2)
            c = 0.1
            u0 = np.exp(-((xx-b[0])**2 + (yy-b[1])**2)/(2*c**2))
        elif prob_dis == 'gaussian2':
            b1 = np.random.normal(0.5, 0.1, 2)
            c1 = 0.1
            u00 = np.exp(-((xx-b1[0])**2 + (yy-b1[1])**2)/(2*c1**2))
            b2 = np.random.normal(0.5, 0.1, 2)
            c2 = 0.1
            u01 = np.exp(-((xx-b2[0])**2 + (yy-b2[1])**2)/(2*c2**2))
            u0 = u00 + u01
        elif prob_dis == 'point':
            b = np.random.randint(0, 64, (2, 1))
            u0 = np.zeros((nel+2, nel+2))
            u0[b[0], b[1]] = 1
        else:
            TypeError('Model is not defined')   
            
        u0[0, :] = 0
        u0[-1, :] = 0
        u0[:, 0] = 0
        u0[:, -1] = 0
        u = u0.copy()
        
        nsteps = 101
        for m in range(nsteps):
            np.save(case_dir + f'/u{m}.npy', u0[1:-1,1:-1])
            u0, u = propagate(u0, u, D, dt, dx2, dy2)
#             u0, u = propagate2(u0, u, D, dt, dx2)

        print('Simulation of case {} is completed!'.format(i))

    print('-'*37+'\n')


if __name__ == '__main__':
    diffusionSim(0.0003, 3, 'uniform', 'data')
    
    
    