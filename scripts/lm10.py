# coding: utf-8

""" Chaos in the Law & Majewski 2010 potential? """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from astropy.constants import G
import astropy.units as u
from scipy.signal import argrelmin
from streamteam.integrate import DOPRI853Integrator
from streamteam.dynamics import lyapunov
from streams.potential._lm10_acceleration import lm10_acceleration, lm10_potential

# Create logger
logger = logging.getLogger(__name__)

plot_path = "plots/lm10"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# unit system
usys = [u.kpc, u.Myr, u.radian, u.M_sun]
_G = G.decompose(usys).value

# standard prolate:
potential_params = [1.38, 1.36, 1.692969, 0.12462565, 1., 12.]

# phase-space position of Sgr today
sgr_w = np.array([19.0149,2.64883,-6.8686,0.23543018,-0.03598748, 0.19917575])

# basically, hamiltons equations
def F(t, X, nparticles, acc, *potential_params):
    x,y,z,px,py,pz = X.T
    dH_dq = lm10_acceleration(X, nparticles, acc, *potential_params)
    return np.hstack((np.array([px, py, pz]).T, dH_dq))

def orbit(w0, potential_params, nsteps=10000, dt=1.):
    nparticles = w0.shape[0]
    acc = np.zeros((nparticles,3))
    integrator = DOPRI853Integrator(F, func_args=(nparticles, acc)+potential_params)
    ts,ws = integrator.run(w0, dt=dt, nsteps=nsteps)
    return ts,ws

def compute_lyapunov(w0, nsteps=10000, potential_params=potential_params):
    nparticles = 2
    acc = np.zeros((nparticles,3))

    integrator = DOPRI853Integrator(F, func_args=(nparticles, acc)+potential_params)
    dt = 1.
    print(nsteps*dt*u.Myr)
    nsteps_per_pullback = 10
    d0 = 1e-5

    LEs, xs = lyapunov(w0, integrator, dt, nsteps,
                       d0=d0, nsteps_per_pullback=nsteps_per_pullback)

    return LEs,xs

def generate_ic_grid(dphi=10*u.deg, drdot=10*u.km/u.s):
    # spacing between IC's in Phi and Rdot
    dphi = dphi.decompose(usys).value
    drdot = drdot.decompose(usys).value
    max_rdot = (400*u.km/u.s).decompose(usys).value
    max_phi = (360*u.deg).decompose(usys).value

    # find the energy of Sgr orbit
    pot = np.zeros((1,))
    T = 0.5*np.sum(sgr_w[3:]**2)
    V = lm10_potential(sgr_w[:3].reshape((1,3)), 1, pot, *potential_params)[0]
    E = T + V

    r = 20. # kpc
    phidot = 0. # rad/Myr
    theta = (90*u.deg).decompose(usys).value # rad

    # T = 0.5*(rdot**2 + thetadot**2/r**2 + phidot**2/(r*np.sin(theta))**2)

    w0s = []
    for rdot in np.arange(0, max_rdot, drdot):
        for phi in np.arange(0, max_phi, dphi):
            X = r*np.cos(phi)*np.sin(theta)
            Y = r*np.sin(phi)*np.sin(theta)
            Z = r*np.cos(theta)
            V = lm10_potential(np.vstack((X,Y,Z)).T, 1, pot, *potential_params)[0]

            # solve for theta dot from energy
            ptheta = r*np.sqrt(2*(E - V) - rdot**2 - phidot**2/(r*np.sin(theta))**2)
            thetadot = ptheta / r**2

            vx = rdot*np.sin(theta)*np.cos(phi) + r*np.cos(theta)*np.cos(phi)*thetadot - r*np.sin(theta)*np.sin(phi)*phidot
            vy = rdot*np.sin(theta)*np.sin(phi) + r*np.cos(theta)*np.sin(phi)*thetadot + r*np.sin(theta)*np.cos(phi)*phidot
            vz = rdot*np.cos(theta) - r*np.sin(theta)*thetadot

            w0 = [X,Y,Z,vx,vy,vz]
            if np.any(np.isnan(w0)): continue
            w0s.append(w0)

    w0s = np.array(w0s)
    return w0s

def potential_grid(nq1=5,nqz=5,nphi=5):
    pps = []
    for q1 in np.append(np.linspace(0.7,1.7,nq1),1.38):
        for qz in np.append(np.linspace(0.7,1.7,nqz),1.36):
            for phi in np.append(np.linspace(0.785,2.356,nphi),1.692969):
                pps.append([q1,qz,phi])
    return np.array(pps)

if __name__ == "__main__":
    # w0s = generate_ic_grid()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(w0s[:,0],w0s[:,1],w0s[:,2],marker='.',linestyle='none')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(w0s[:,3],w0s[:,4],w0s[:,5],marker='.',linestyle='none')
    # plt.show()
    # sys.exit(0)

    # t,w = orbit(sgr_w.reshape((1,6)), potential_params)
    # w0 = np.array([19.,2.,-6.9,0.05,0.0, 0.05]).reshape((1,6))

    # # Vary orbit parameters
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # fig3d = plt.figure(figsize=(10,10))
    # ax3d = fig3d.add_subplot(111, projection='3d')
    # for ii,w0 in enumerate(w0s):
    #     LEs,ws = compute_lyapunov(w0, nsteps=100000)

    #     print("Lyapunov exponent computed")
    #     ax.cla()
    #     ax.semilogy(LEs, marker=None)
    #     fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

    #     ax3d.cla()
    #     ax3d.plot(ws[:,0], ws[:,1], ws[:,2], marker=None)
    #     fig3d.savefig(os.path.join(plot_path,'orbit_{}.png'.format(ii)))

    # Vary potential parameters
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    fig3d = plt.figure(figsize=(10,10))
    ax3d = fig3d.add_subplot(111, projection='3d')
    for ii,pp in enumerate(potential_grid(nq1=5,nqz=5,nphi=5)):
        pparams = potential_params
        pparams[0] = pp[0]
        pparams[1] = pp[1]
        pparams[3] = pp[2]
        LEs,ws = compute_lyapunov(sgr_w, nsteps=10000,
                                  potential_params=tuple(pparams))

        print("Lyapunov exponent computed")
        ax.cla()
        ax.set_title("q1={}, qz={}, phi={}".format(*pp))
        ax.semilogy(LEs, marker=None)
        fig.savefig(os.path.join(plot_path,'le_{}.png'.format(ii)))

        ax3d.cla()
        ax3d.set_title("q1={}, qz={}, phi={}".format(*pp))
        ax3d.plot(ws[:,0], ws[:,1], ws[:,2], marker=None)
        fig3d.savefig(os.path.join(plot_path,'orbit_{}.png'.format(ii)))

    sys.exit(0)

    t,w = orbit(sgr_w.reshape((1,6)), potential_params)

    plt.figure(figsize=(10,10))
    plt.plot(w[:,0,0], w[:,0,2], marker=None, linestyle='-')
    plt.savefig(os.path.join(plot_path, "sgr.png"))
