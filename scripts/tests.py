# coding: utf-8

""" Compute Lyapunov exponents """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmin
from streamteam.integrate import LeapfrogIntegrator, RK5Integrator, DOPRI853Integrator
from streams.potential import PointMassPotential, MiyamotoNagaiPotential
from streams.potential.lm10 import LawMajewski2010
from streamteam.dynamics import lyapunov

plot_path = "plots"

def RRdot_surface_of_section(xs, z_slice=0.):
    R,z,vR,vz = xs.T
    w, = argrelmin(np.abs(z-z_slice))

    ix = np.zeros_like(z).astype(bool)
    ix[w] = True
    ix &= vz > 0.

    # plt.clf()
    # plt.plot(np.abs(z))
    # plt.plot(w, np.abs(z)[w], marker='o', color='r')
    # plt.show()
    # sys.exit(0)

    return R[ix], vR[ix]

def pendulum():
    # Just a test of the Lyapunov exponent calculation

    # CHAOTIC
    # A = 0.07
    # omega_d = 0.75
    # x0 = np.array([3.,0.,0.])
    # ext = "chaotic"

    # REGULAR
    A = 0.055
    omega_d = 0.7
    x0 = np.array([1.,0.,0.])
    ext = "regular"

    def F(t,x):
        q,p,z = x.T
        arr = np.array([p, -np.sin(q) + A*np.cos(omega_d*t), p*0.]).T
        return arr

    integrator = DOPRI853Integrator(F)
    nsteps = 100000
    dt = 0.1
    nsteps_per_pullback = 10
    d0 = 1e-5

    LEs, xs = lyapunov(x0, integrator, dt, nsteps,
                       d0=d0, nsteps_per_pullback=nsteps_per_pullback)

    print("Lyapunov exponent computed")
    plt.clf()
    plt.semilogy(LEs, marker=None)
    plt.savefig("pend_le_{}.png".format(ext))

    plt.clf()
    plt.plot(xs[:,0],xs[:,1],marker=None)
    plt.savefig(os.path.join(plot_path,"pend.png"))

def point_mass():
    GM = (G * (1.*u.M_sun)).decompose([u.au,u.M_sun,u.year,u.radian]).value

    def F(t,x):
        x,y,px,py = x.T
        a = -GM/(x*x+y*y)**1.5
        return np.array([px, py, x*a, y*a]).T

    x0 = np.array([1.0, 0.0, 0.0, 2*np.pi]) # Earth

    integrator = DOPRI853Integrator(F)
    nsteps = 10000
    dt = 0.01
    nsteps_per_pullback = 10
    d0 = 1e-5

    LEs, xs = lyapunov(x0, integrator, dt, nsteps,
                       d0=d0, nsteps_per_pullback=nsteps_per_pullback)

    print("Lyapunov exponent computed")
    plt.clf()
    plt.semilogy(LEs, marker=None)
    plt.savefig("point_mass_le.png")

    plt.clf()
    plt.plot(xs[:,0],xs[:,1],marker=None)
    plt.savefig(os.path.join(plot_path,"point_mass.png"))


def pal5():
    from streams import usys

    r0 = np.array([[8.312877511,0.242593717,16.811943627]])
    v0 = ([[-52.429087,-96.697363,-8.156130]]*u.km/u.s).decompose(usys).value

    x0 = np.append(r0,v0)
    acc = np.zeros((2,3))
    potential = LawMajewski2010()

    def F(t,X):
        x,y,z,px,py,pz = X.T
        dH_dq = potential._acceleration_at(X[...,:3], 2, acc)
        return np.hstack((np.array([px, py, pz]).T, dH_dq))

    integrator = DOPRI853Integrator(F)
    nsteps = 100000
    dt = 0.1
    print(nsteps*dt*u.Myr)
    nsteps_per_pullback = 10
    d0 = 1e-5

    LEs, xs = lyapunov(x0, integrator, dt, nsteps,
                       d0=d0, nsteps_per_pullback=nsteps_per_pullback)

    print("Lyapunov exponent computed")
    plt.clf()
    plt.semilogy(LEs, marker=None)
    plt.savefig('pal5_le.png')

    plt.clf()
    plt.plot(np.sqrt(xs[:,0]**2+xs[:,1]**2), xs[:,2], marker=None)
    plt.savefig('pal5_orbit.png')

def lm10_grid():
    from streams import usys
    acc = np.zeros((2,3))
    potential = LawMajewski2010()

    def F(t,X):
        x,y,z,px,py,pz = X.T
        dH_dq = potential._acceleration_at(X[...,:3], 2, acc)
        return np.hstack((np.array([px, py, pz]).T, dH_dq))

    nsteps = 1000000
    dt = 0.1
    print(nsteps*dt*u.Myr)
    nsteps_per_pullback = 10
    d0 = 1e-5

    # will vary x,vx
    y = 0.
    z = 0.
    vy = 0.
    E = (6E4*(u.km/u.s)**2).decompose(usys).value

    vx_max = (300*u.km/u.s).decompose(usys).value
    for x in np.linspace(10., 30., 10): # kpc
        for vx in np.linspace(0., vx_max, 10): # kpc/Myr
            r0 = np.array([[x,y,z]])
            V = potential._value_at(r0)
            vz = np.squeeze(np.sqrt(2*(E - V) - vx**2 - vy**2))
            print(x, y, z, vx, vy, vz, V)
            v0 = np.array([[vx,vy,vz]])
            x0 = np.append(r0,v0)

            integrator = DOPRI853Integrator(F)

            LEs, xs = lyapunov(x0, integrator, dt, nsteps,
                               d0=d0,
                               nsteps_per_pullback=nsteps_per_pullback)

            print("Lyapunov exponent computed")
            plt.clf()
            plt.semilogy(LEs, marker=None)
            plt.savefig('lm10_le_x{:.1f}_vx{:.3f}.png'.format(x,vx))

            plt.clf()
            plt.plot(xs[:,0], xs[:,1], marker=None)
            plt.savefig('lm10_orbit_x{:.1f}_vx{:.3f}.png'.format(x,vx))


if __name__ == "__main__":
    np.random.seed(42)
    #pendulum()
    #point_mass()


    #zotos('box')
    zotos('2-1-banana')
    #zotos('1-1-linear')
    zotos('3-2-boxlet')
    zotos('4-3-boxlet')
    zotos('8-5-boxlet')
    zotos('13-8-boxlet')
    zotos('chaos')
    # zotos('chaos2')
    # zotos('chaos3')

    # pal5()

    # lm10_grid()

    # zotos_ball('4-3-box')
    # zotos_ball('2-1-box')
    # zotos_ball('8-5-box')
    # zotos_ball('3-2-box')
    # zotos_ball('chaos')
    # zotos_ball('straddle')

"""
ffmpeg -r 10 -i RZ_zotos_ball_%05d.png -codec h264 -r 10 -pix_fmt yuv420p 0RZ.mp4
ffmpeg -r 10 -i XZ_zotos_ball_%05d.png -codec h264 -r 10 -pix_fmt yuv420p 0XZ.mp4
"""
