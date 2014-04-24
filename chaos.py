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
from streamteam.integrate import LeapfrogIntegrator, RK5Integrator, DOPRI853Integrator
from streams.potential import PointMassPotential, MiyamotoNagaiPotential
from streams.potential.lm10 import LawMajewski2010

plot_path = ""

def lyapunov(x0, integrator, dt, nsteps, d0=1e-5, nsteps_per_pullback=10):
    """ Compute the Lyapunov exponent of an orbit from initial conditions
        (q0,p0).
    """

    niter = nsteps // nsteps_per_pullback
    ndim = x0.size

    # define an offset vector to start the offset orbit on
    d0_vec = np.zeros_like(x0)
    d0_vec[0] = d0

    x_offset = x0 + d0_vec
    x_i = np.vstack((x0,x_offset))

    full_x = np.zeros((nsteps+1,ndim))
    full_x[0] = x0

    LEs = np.zeros(niter)
    ts = np.zeros_like(LEs)
    time = 0.
    for i in range(1,niter+1):
        ii = i * nsteps_per_pullback

        tt,xx = integrator.run(x_i, dt=dt, nsteps=nsteps_per_pullback)
        time += tt[-1]

        main_x = xx[-1,0]
        d1 = xx[-1,1] - main_x
        d1_mag = np.linalg.norm(d1)
        LEs[i-1] = np.log(d1_mag/d0)
        ts[i-1] = time

        x_offset = xx[-1,0] + d0 * d1 / d1_mag
        x_i = np.vstack((xx[-1,0],x_offset))

        full_x[(i-1)*nsteps_per_pullback+1:ii+1] = xx[1:,0]

    LEs = np.array([LEs[:ii].sum()/ts[ii-1] for ii in range(1,niter)])

    return LEs, full_x

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

def zotos_potential(r, G, Md, alpha, b, h, Mn, cn, v0, beta, ch):
    x,y,z = r.T
    Rsq = x**2 + y**2

    Vd = -G*Md / np.sqrt(b**2 + Rsq + (alpha + np.sqrt(h**2+z**2))**2)
    Vn = -G*Mn / np.sqrt(Rsq+z**2+cn**2)
    Vh = v0**2/2 * np.log(Rsq+beta*z**2 + ch**2)

    return Vd + Vn + Vh

def zotos_acceleration(r, G, Md, alpha, b, h, Mn, cn, v0, beta, ch):
    x,y,z = r.T

    Rsq = x**2 + y**2
    tmp1 = -G*Md/(b**2 + Rsq + (alpha + np.sqrt(h**2 + z**2))**2)**1.5
    tmp2 = -G*Mn/(cn**2 + Rsq + z**2)**1.5
    tmp3 = -v0**2/(beta*z**2 + ch**2 + Rsq)

    dx = x*tmp1 + x*tmp2 + x*tmp3
    dy = y*tmp1 + y*tmp2 + y*tmp3
    dz = z*tmp1*(alpha + np.sqrt(h**2 + z**2))/np.sqrt(h**2 + z**2) + z*tmp2 + beta*z*tmp3

    return np.array([dx,dy,dz]).T

def zotos(orbit_type):
    from streams import usys
    from astropy.constants import G
    G = G.decompose(usys).value

    # Parameter choices
    # standard prolate:
    # params = (G, 1.63E11, 3., 6., 0.2,
    #           5.8E9, 0.25,
    #           0.204542433, 0.5, 8.5)
    # standard oblate:
    params = (G, 1.63E11, 3., 6., 0.2,
              5.8E9, 0.25,
              0.204542433, 1.5, 8.5)

    # 4:3 boxlet?
    if orbit_type == "4-3-box":
        r0 = np.array([2.7,0.,0.])
        vx = (290*u.km/u.s).decompose(usys).value

    # 2:1 boxlet?
    elif orbit_type == "2-1-box":
        r0 = np.array([6.,0.,0.])
        vx = 0.

    # 8:5 boxlet?
    elif orbit_type == "8-5-box":
        r0 = np.array([10.,0.,0.])
        vx = 0.

    # 3:2 boxlet?
    elif orbit_type == "3-2-box":
        r0 = np.array([8.1,0.,0.])
        vx = (175*u.km/u.s).decompose(usys).value

    # chaos!
    elif orbit_type == "chaos":
        r0 = np.array([10.,0.,0.])
        vx = (180*u.km/u.s).decompose(usys).value

    # chaos!
    elif orbit_type == "chaos2":
        r0 = np.array([2.,0.,0.])
        vx = (360.*u.km/u.s).decompose(usys).value

    # chaos!
    elif orbit_type == "chaos3":
        r0 = np.array([5.,0.,0.])
        vx = (200.*u.km/u.s).decompose(usys).value

    else:
        return

    E = (600*100*(u.km/u.s)**2).decompose(usys).value
    V = zotos_potential(r0, *params)
    Lz = (10.*10.*u.km*u.kpc/u.s).decompose(usys).value # typo in paper? km/kpc instead of km*kpc

    R = np.sqrt(r0[...,0]**2 + r0[...,1]**2)
    vz = np.squeeze(np.sqrt(2*E - 2*V - Lz**2/R**2 - vx**2))
    v0 = np.array([vx,0.,vz])

    x0 = np.append(r0,v0)

    def F(t,X):
        x,y,z,px,py,pz = X.T
        dH_dq = zotos_acceleration(X[...,:3], *params)
        return np.hstack((np.array([px, py, pz]).T, dH_dq))

    integrator = DOPRI853Integrator(F, func_args=params)
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
    plt.savefig('zotos_le_{}.png'.format(orbit_type))

    plt.clf()
    plt.plot(np.sqrt(xs[:,0]**2+xs[:,1]**2), xs[:,2], marker=None)
    plt.savefig('zotos_orbit_{}.png'.format(orbit_type))

def zotos_ball(orbit_type):
    from streams import usys
    from astropy.constants import G
    G = G.decompose(usys).value

    dt = 0.1
    nsteps = 50000
    nstars = 1000

    # standard oblate:
    params = (G, 1.63E11, 3., 6., 0.2,
              5.8E9, 0.25,
              0.204542433, 1.5, 8.5)

    # determine initial conditions
    # 4:3 boxlet?
    if orbit_type == "4-3-box":
        x0 = np.array([[11.6,0.,0.]])
        vx = 0.

    # 2:1 boxlet?
    elif orbit_type == "2-1-box":
        x0 = np.array([[6.,0.,0.]])
        vx = 0.

    # 8:5 boxlet?
    elif orbit_type == "8-5-box":
        x0 = np.array([[10.,0.,0.]])
        vx = 0.

    # 3:2 boxlet?
    elif orbit_type == "3-2-box":
        x0 = np.array([[8.1,0.,0.]])
        vx = (175*u.km/u.s).decompose(usys).value

    # chaos!
    elif orbit_type == "chaos":
        x0 = np.array([[10.6,0.,0.]])
        vx = 0.

    # near edge
    elif orbit_type == "straddle":
        x0 = np.array([[0.,8.,0.]])
        vx = (120*u.km/u.s).decompose(usys).value

    else:
        return

    E = (600*100*(u.km/u.s)**2).decompose(usys).value
    V = zotos_potential(x0, *params)
    Lz = (10.*10.*u.km*u.kpc/u.s).decompose(usys).value # typo in paper? km/kpc instead of km*kpc

    R = np.sqrt(x0[...,0]**2 + x0[...,1]**2)
    vz = np.squeeze(np.sqrt(2*E - 2*V - Lz**2/R**2 - vx**2))
    v0 = np.array([[vx,0.,vz]])
    print(E, V, Lz**2/R**2, vx, vz)

    all_x0 = np.random.normal(x0, 0.2, size=(nstars,3))
    all_v0 = np.random.normal(v0, (10*u.km/u.s).decompose(usys).value, size=(nstars,3))
    all_x0 = np.vstack((x0,all_x0))
    all_v0 = np.vstack((v0,all_v0))

    integrator = LeapfrogIntegrator(zotos_acceleration, func_args=params)
    t,q,p = integrator.run(all_x0.copy(), all_v0.copy(),
                           dt=dt, nsteps=nsteps)

    if not os.path.exists(orbit_type):
        os.mkdir(orbit_type)

    plt.figure(figsize=(10,10))

    ii = 0
    for jj in range(nsteps):
        if jj % 100 != 0:
            continue

        plt.clf()
        plt.plot(np.sqrt(q[:,0,0]**2 + q[:,0,1]**2), q[:,0,2],
                marker=None, linestyle='-', alpha=0.5, color='#3182bd')
        plt.plot(np.sqrt(q[jj,1:,0]**2 + q[jj,1:,1]**2), q[jj,1:,2],
                marker='.', linestyle='none', alpha=0.65)
        plt.xlabel("R")
        plt.ylabel("Z")
        plt.xlim(0., 18)
        plt.ylim(-15, 15)
        plt.savefig(os.path.join(orbit_type, 'RZ_zotos_ball_{:05d}.png'.format(ii)))

        plt.clf()
        plt.plot(q[:,0,0], q[:,0,2],
                marker=None, linestyle='-', alpha=0.5, color='#3182bd')
        plt.plot(q[jj,1:,0], q[jj,1:,2],
                marker='.', linestyle='none', alpha=0.65)
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.xlim(-15., 15)
        plt.ylim(-15, 15)
        plt.savefig(os.path.join(orbit_type, 'XZ_zotos_ball_{:05d}.png'.format(ii)))

        ii += 1

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
            print(x, y, z, vx, vy, vz)
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
            plt.plot(np.sqrt(xs[:,0]**2+xs[:,1]**2), xs[:,2], marker=None)
            plt.savefig('lm10_orbit_x{:.1f}_vx{:.3f}.png'.format(x,vx))


if __name__ == "__main__":
    np.random.seed(42)
    #pendulum()
    #point_mass()
    # zotos('4-3-box')
    # zotos('chaos')
    # zotos('2-1-box')
    # zotos('8-5-box')
    # zotos('3-2-box')
    # zotos('chaos2')
    # zotos('chaos3')

    # pal5()

    lm10_grid()

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
