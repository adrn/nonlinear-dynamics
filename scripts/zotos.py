# coding: utf-8

""" Reproducing figures from Zotos 2014 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G
import astropy.units as u
from scipy.signal import argrelmin
from streamteam.integrate import DOPRI853Integrator
from streamteam.dynamics import lyapunov

# Create logger
logger = logging.getLogger(__name__)

plot_path = "plots"

usys = [u.kpc, u.Myr, u.radian, u.M_sun]
_G = G.decompose(usys).value
# standard prolate:
prolate_params = (_G, 1.63E11, 3., 6., 0.2,
                  5.8E9, 0.25,
                  0.204542433, 0.5, 8.5)

# standard oblate:
oblate_params = (_G, 1.63E11, 3., 6., 0.2,
                 5.8E9, 0.25,
                 0.204542433, 1.5, 8.5)

def zotos_potential(R, z, G, Md, alpha, b, h, Mn, cn, v0, beta, ch):
    Rsq = R*R

    Vd = -G*Md / np.sqrt(b**2 + Rsq + (alpha + np.sqrt(h**2+z**2))**2)
    Vn = -G*Mn / np.sqrt(Rsq+z**2+cn**2)
    Vh = v0**2/2 * np.log(Rsq+beta*z**2 + ch**2)

    return Vd + Vn + Vh

def zotos_acceleration(R, z, Lz, G, Md, alpha, b, h, Mn, cn, v0, beta, ch):
    Rsq = R*R

    tmp1 = -G*Md/(b**2 + Rsq + (alpha + np.sqrt(h**2 + z**2))**2)**1.5
    tmp2 = -G*Mn/(cn**2 + Rsq + z**2)**1.5
    tmp3 = -v0**2/(beta*z**2 + ch**2 + Rsq)

    dR = R*tmp1 + R*tmp2 + R*tmp3 + Lz**2/R**3
    dz = z*tmp1*(alpha + np.sqrt(h**2 + z**2))/np.sqrt(h**2 + z**2) + z*tmp2 + beta*z*tmp3

    return np.array([dR,dz]).T

# basically, hamiltons equations
def F(t,X,*params):
    R,z,pR,pz = X.T
    dH_dq = zotos_acceleration(R, z, params[0], *params[1:])
    return np.hstack((np.array([pR, pz]).T, dH_dq))

def orbit(w0, potential_params, nsteps=10000, dt=1.):
    integrator = DOPRI853Integrator(F, func_args=potential_params)
    ts,ws = integrator.run(w0, dt=dt, nsteps=nsteps)
    return ts,ws

def compute_lyapunov(orbit_name, halo_shape='oblate'):
    if halo_shape == "prolate":
        params = prolate_params
        raise NotImplementedError()
    elif halo_shape == "oblate":
        params = oblate_params
    else:
        raise ValueError("Invalid halo shape.")

    integrator = DOPRI853Integrator(F, func_args=params)
    nsteps = 10000
    dt = 1.
    print(nsteps*dt*u.Myr)
    nsteps_per_pullback = 10
    d0 = 1e-5

    LEs, xs = lyapunov(x0, integrator, dt, nsteps,
                       d0=d0, nsteps_per_pullback=nsteps_per_pullback)

    print("Lyapunov exponent computed")
    plt.figure(figsize=(10,10))
    plt.clf()
    plt.semilogy(LEs, marker=None)
    plt.savefig('zotos_le_{}.png'.format(orbit_type))

    plt.clf()
    plt.plot(xs[:,0], xs[:,1], marker=None)
    plt.xlim(0,15)
    plt.ylim(-15,15)
    plt.savefig('zotos_orbit_{}.png'.format(orbit_type))

def initial_conditions(orbit_name, halo_shape='oblate'):

    if halo_shape == "prolate":
        params = prolate_params
        raise NotImplementedError()
    elif halo_shape == "oblate":
        params = oblate_params
    else:
        raise ValueError("Invalid halo shape.")

    if orbit_name == "box":
        r0 = np.array([1.3,0.])
        vR = 0.

    elif orbit_name == "2-1-banana":
        r0 = np.array([6.00338292,0.])
        vR = 0.

    elif orbit_name == "1-1-linear":
        r0 = np.array([5.046266,0.])
        vR = (30.92524*10*u.km/u.s).decompose(usys).value

    elif orbit_name == "3-2-boxlet":
        r0 = np.array([1.50953,0.])
        vR = (20.20790*10*u.km/u.s).decompose(usys).value

    elif orbit_name == "4-3-boxlet":
        r0 = np.array([11.70795,0.])
        vR = 0.

    elif orbit_name == "8-5-boxlet":
        r0 = np.array([9.971466,0.])
        vR = 0.

    elif orbit_name == "13-8-boxlet":
        r0 = np.array([9.5316067,0.])
        vR = 0.

    elif orbit_name == "chaos":
        r0 = np.array([0.18,0.])
        vR = 0.

    elif orbit_name == "chaos2":
        r0 = np.array([5.,0.])
        vR = (22.*10*u.km/u.s).decompose(usys).value

    R,z = r0[...,:2].T
    E = (600*100*(u.km/u.s)**2).decompose(usys).value
    V = zotos_potential(R, z, *params)
    Lz = (10.*10.*u.km*u.kpc/u.s).decompose(usys).value # typo in paper? km/kpc instead of km*kpc

    vz = np.squeeze(np.sqrt(2*(E - V) - Lz**2/R**2 - vR**2))
    v0 = np.array([vR,vz])

    w0 = np.append(r0,v0)
    return w0, Lz

def zotos_ball(progenitor_orbit_name):
    dt = 1.
    nsteps = 5500
    nstars = 1000

    progenitor_w0,Lz = initial_conditions(progenitor_orbit_name, 'oblate')
    print(progenitor_w0)
    all_w0 = np.random.normal(progenitor_w0, [0.2,0.2,0.014,0.014], size=(nstars,4))
    all_w0 = np.vstack((progenitor_w0, all_w0))

    print(nsteps*dt*u.Myr)
    t,w = orbit(all_w0, (Lz,)+oblate_params, nsteps=nsteps, dt=dt)
    print("finished integrating")

    path = os.path.join(plot_path, progenitor_orbit_name)
    if not os.path.exists(path):
        os.mkdir(path)

    plt.figure(figsize=(10,10))

    ii = 0
    for jj in range(nsteps):
        if jj % 10 != 0:
            continue

        plt.clf()
        plt.plot(w[:,0,0], w[:,0,1],
                marker=None, linestyle='-', alpha=0.5, color='#3182bd')
        plt.plot(w[jj,1:,0], w[jj,1:,1],
                marker='.', linestyle='none', alpha=0.65)
        plt.xlabel("R")
        plt.ylabel("Z")
        plt.xlim(0., 18)
        plt.ylim(-15, 15)
        plt.savefig(os.path.join(path, 'RZ_zotos_ball_{:05d}.png'.format(ii)))

        ii += 1

def generate_ic_grid(dR=0.1*u.kpc, dRdot=5.*u.km/u.s):
    # spacing between IC's in R and Rdot
    dR = dR.decompose(usys).value
    dRdot = dRdot.decompose(usys).value
    max_Rdot = (50*10*u.km/u.s).decompose(usys).value
    max_R = (15*u.kpc).decompose(usys).value

    # from the paper
    E = (600*100*(u.km/u.s)**2).decompose(usys).value
    Lz = (10.*10.*u.km*u.kpc/u.s).decompose(usys).value # typo in paper? km/kpc instead of km*kpc
    z = 0.
    params = oblate_params

    w0s = []
    for R in np.arange(0, max_R, dR):
        # zero velocity curve
        V = zotos_potential(R, z, *params)
        ZVC_Rdot = np.squeeze(np.sqrt(2*(E-V) - Lz**2/R**2))
        for Rdot in np.arange(0, max_Rdot, dRdot):
            if Rdot > ZVC_Rdot or R < 0.2 or R >= 13: continue
            zdot = np.squeeze(np.sqrt(2*(E - V) - Lz**2/R**2 - Rdot**2))
            w0 = [R,z,Rdot,zdot]
            w0s.append(w0)
    w0s = np.array(w0s)
    return w0s, Lz

def RRdot_surface_of_section(ws, z_slice=0.):
    Rs = np.array([])
    vRs = np.array([])
    for i in range(ws.shape[1]):
        R,z,vR,vz = ws[:,i].T
        k, = argrelmin(np.abs(z-z_slice))

        ix = np.zeros_like(z).astype(bool)
        ix[k[2:]] = True
        ix &= vz > 0.

        Rs = np.append(Rs,R[ix])
        vRs = np.append(vRs, vR[ix])

    return Rs, vRs

def animate_orbit(orbit_name):

    # orbit in 3D
    w0, Lz = initial_conditions(orbit_name, halo_shape='oblate')
    t,w = orbit(w0, (Lz,)+oblate_params, nsteps=5000, dt=1.)
    print("orbit integrated")

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(xlim=(0., 15), ylim=(-15, 15))
    particle, = ax.plot([], [], linestyle='none', markersize=10,
                        marker='o', markeredgecolor='none')
    line, = ax.plot([], [], linestyle='-', marker=None, linewidth=2., alpha=0.75, zorder=-1)
    ax.set_xlabel("R [kpc]", fontsize=22)
    ax.set_ylabel("Z [kpc]", fontsize=22)

    # initialization function: plot the background of each frame
    def init():
        line.set_data(w[...,0], w[...,1])
        particle.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        particle.set_data(w[i*4,0,0],w[i*4,0,1])
        return particle,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(w)//4, interval=20, blit=True)

    anim.save(os.path.join(plot_path,orbit_name,'animate.mp4'),
              fps=15, extra_args=['-codec', 'h264', '-pix_fmt', 'yuv420p'])

if __name__ == "__main__":

    # ball of particles
    # zotos_ball("chaos2")
    # zotos_ball("4-3-boxlet")
    # zotos_ball("3-2-boxlet")
    # zotos_ball("1-1-linear")
    # zotos_ball("2-1-banana")
    # zotos_ball("8-5-boxlet")
    # sys.exit(0)

    animate_orbit("3-2-boxlet")
    animate_orbit("4-3-boxlet")
    animate_orbit("8-5-boxlet")
    animate_orbit("chaos")
    animate_orbit("chaos2")
    sys.exit(0)

    # w0,Lz = initial_conditions("4-3-boxlet", "oblate")
    fname = "R_Rdot.npy"

    if not os.path.exists(fname):
        w0,Lz = generate_ic_grid(dR=0.1*u.kpc, dRdot=5*u.km/u.s)
        t,w = orbit(w0, (Lz,)+oblate_params, nsteps=10000)
        np.save(fname, w)
    else:
        w = np.load(fname)

    Rs, vRs = RRdot_surface_of_section(w)
    vRs = (vRs*u.kpc/u.Myr).to(u.km/u.s).value

    plt.figure(figsize=(10,10))
    plt.plot(Rs, vRs/10., marker=',', alpha=0.75, linestyle='none')
    plt.xlim(0,14)
    plt.ylim(0,50)
    plt.savefig(os.path.join(plot_path, "SOS.png"))

    plt.clf()
    plt.hexbin(Rs, vRs, cmap=plt.cm.Blues)
    plt.axis([Rs.min(), Rs.max(), vRs.min(), vRs.max()])
    plt.savefig(os.path.join(plot_path, "SOS_bin.png"))

    #plt.show()
    #plt.savefig('zotos_orbit_{}.png'.format(orbit_type))


"""

    R,vR = RRdot_surface_of_section(xs, z_slice=0.)
    plt.clf()
    plt.plot(R, (vR*u.kpc/u.Myr).to(u.km/u.s).value/10., marker='.', linestyle='none')
    plt.xlim(0,15)
    plt.ylim(-45, 45)
    plt.savefig('zotos_sos_{}.png'.format(orbit_type))

    return
"""
