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
from streamteam.integrate import LeapfrogIntegrator, RK5Integrator
from streams.potential import PointMassPotential, MiyamotoNagaiPotential
from streams.potential.lm10 import LawMajewski2010

plot_path = ""

def lyapunov(q0, p0, integrator, dt, nsteps, d0=1e-8, nsteps_per_pullback=10):
    """ Compute the Lyapunov exponent of an orbit from initial conditions
        (q0,p0).
    """

    niter = nsteps//nsteps_per_pullback
    if int(nsteps / nsteps_per_pullback) != niter:
        raise ValueError("BORKED")

    # define an offset vector to start the offset orbit on
    dq = np.random.uniform(-1,1,size=(q0.shape))
    dq = d0*dq/np.squeeze(np.linalg.norm(dq))
    dp = np.zeros_like(p0)

    q_offset = q0 + dq
    p_offset = p0 + dp

    q_init = np.vstack((q0,q_offset))
    p_init = np.vstack((p0,p_offset))

    LEs = np.zeros(niter)
    ts = np.zeros_like(LEs)
    time = 0.
    for i in range(1,niter+1):
        ii = i * nsteps_per_pullback

        tt,qq,pp = integrator.run(q_init, p_init,
                                  dt=dt, nsteps=nsteps_per_pullback)
        time += tt[-1]

        main_q = qq[-1,0]
        main_p = pp[-1,0]

        q_offset = qq[-1,1]
        p_offset = pp[-1,1]

        d1_q = main_q - q_offset
        d1_p = main_p - p_offset
        d1 = np.append(d1_q,d1_p)
        d1_mag = np.linalg.norm(d1)

        LEs[i-1] = np.log(d1_mag/d0)
        ts[i-1] = time

        if d1_mag > 0:
            q_offset = main_q + d0 * d1[:3] / d1_mag
            p_offset = main_p + d0 * d1[3:] / d1_mag
        else:
            q_offset = main_q
            p_offset = main_p

        q_init = np.vstack((main_q,q_offset))
        p_init = np.vstack((main_p,p_offset))

    LEs = np.array([LEs[:ii].sum()/ts[ii-1] for ii in range(1,niter)])
    return LEs

def pendulum():
    # Just a test of the Lyapunov exponent calculation

    # CHAOTIC
    A = 0.07
    omega_d = 0.75
    q0,p0 = np.array([3.]), np.array([0.])
    ext = "chaotic"

    # REGULAR
    # A = 0.055
    # omega_d = 0.7
    # q0,p0 = np.array([1.]), np.array([0.])
    # ext = "regular"

    def F(t,x):
        q,p = x.T
        return np.array([p,-np.sin(q) + A*np.cos(omega_d*t)]).T.copy()

    integrator = RK5Integrator(F)
    nsteps = 100000
    # LE = lyapunov(np.array([3.]), np.array([0.]),
    #               integrator, dt=0.1, nsteps=nsteps,
    #               d0=1E-8, nsteps_per_pullback=10)

    nsteps_per_pullback = 1
    niter = nsteps//nsteps_per_pullback

    # define an offset vector to start the offset orbit on
    d0 = 1e-3
    q_offset = q0 + d0
    p_offset = p0

    q_init = np.vstack((q0,q_offset))
    p_init = np.vstack((p0,p_offset))

    LEs = np.zeros(niter)
    ts = np.zeros_like(LEs)
    time = 0.
    for i in range(1,niter+1):
        ii = i * nsteps_per_pullback

        tt,qq,pp = integrator.run(q_init, p_init,
                                  dt0=0.2, nsteps=nsteps_per_pullback)
        time += tt[-1]

        d1 = np.array([qq[-1,1] - qq[-1,0],
                       pp[-1,1] - pp[-1,0]])
        d1_mag = np.linalg.norm(d1)
        LEs[i-1] = np.log(d1_mag/d0)
        ts[i-1] = time

        q_offset = qq[-1,0] + d0 * d1[0] / d1_mag
        p_offset = pp[-1,0] + d0 * d1[1] / d1_mag

        q_init = np.vstack((qq[-1,0],q_offset))
        p_init = np.vstack((pp[-1,0],p_offset))

    LEs = np.array([LEs[:ii].sum()/ts[ii-1] for ii in range(1,niter)])

    print("Lyapunov exponent computed")
    plt.clf()
    plt.semilogy(LEs, marker=None)
    plt.savefig("pend_le_{}.png".format(ext))

    #fig = plot(ts, qs, ps)
    # fig,ax = plt.subplots(1,1)
    # #ax.plot(ts,qs[:,0,0],marker=None)
    # ax.plot(qs[:,0,0],ps[:,0,0],marker=None)
    # fig.savefig(os.path.join(plot_path,"rk5_pend.png"))

def point_mass():
    dt = 0.01
    nsteps = 100000

    usys = [u.au, u.M_sun, u.yr]
    X = 1.
    Y = 0.
    Z = 0.
    x0 = np.array([[X,Y,Z]])

    Vx = 0.
    Vy = 2.*np.pi
    Vz = 0.
    v0 = np.array([[Vx,Vy,Vz]])

    potential = PointMassPotential(units=usys, m=1*u.M_sun,
                                   r_0=[0.,0.,0.]*u.au)
    integrator = LeapfrogIntegrator(potential._acceleration_at)

    LE = lyapunov(x0, v0, integrator,
                  dt=dt, nsteps=nsteps,
                  d0=1E-8, nsteps_per_pullback=1)

    print("Lyapunov exponent computed")
    plt.clf()
    plt.semilogy(LE, marker=None)
    plt.savefig("pt_mass_le.png")

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

    dt = 0.05
    nsteps = 1000000
    print(nsteps*dt*u.Myr)

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
        x0 = np.array([[2.7,0.,0.]])
        vx = (290*u.km/u.s).decompose(usys).value

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
        x0 = np.array([[0.,10.,0.]])
        vx = (180*u.km/u.s).decompose(usys).value

    else:
        return

    E = (600*100*(u.km/u.s)**2).decompose(usys).value
    V = zotos_potential(x0, *params)
    Lz = (10.*10.*u.km*u.kpc/u.s).decompose(usys).value # typo in paper? km/kpc instead of km*kpc

    R = np.sqrt(x0[...,0]**2 + x0[...,1]**2)
    vz = np.squeeze(np.sqrt(2*E - 2*V - Lz**2/R**2 - vx**2))
    v0 = np.array([[vx,0.,vz]])

    integrator = LeapfrogIntegrator(zotos_acceleration, func_args=params)
    LE = lyapunov(x0, v0, integrator,
                  dt=dt, nsteps=nsteps,
                  d0=1E-5, nsteps_per_pullback=1)

    print("Lyapunov exponent computed")
    plt.clf()
    plt.semilogy(LE, marker=None)
    plt.savefig('zotos_le_{}.png'.format(orbit_type))

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

if __name__ == "__main__":
    np.random.seed(42)
    #pendulum()

    #point_mass()
    zotos('chaos')
    zotos('4-3-box')

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