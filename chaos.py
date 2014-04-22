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
from streamteam.integrate import LeapfrogIntegrator
from streams.potential import PointMassPotential, MiyamotoNagaiPotential
from streams.potential.lm10 import LawMajewski2010

def lyapunov(q, p, dt, integrator, d0=1e-8):
    q_offset = q[0] + np.array([[d0,0.,0.]])
    p_offset = p[0] + np.array([[0.,0.,0.]])
    nsteps = q.shape[0]

    LEs = []
    t = 0.
    for ii in range(nsteps-1):
        t += (ii+1)*dt
        t_offset,qq,pp = integrator.run(q_offset.copy(), p_offset.copy(),
                                        dt=dt, nsteps=2)

        d1_q = qq[1,0] - q[ii+1,0]
        d1_p = pp[1,0] - p[ii+1,0]
        d1 = np.append(d1_q,d1_p)
        d1_mag = np.linalg.norm(d1)

        LEs.append(np.log(d1_mag/d0))

        q_offset = q[ii+1] + d0 * d1[:3] / d1_mag
        p_offset = p[ii+1] + d0 * d1[3:] / d1_mag

    LEs = np.array(LEs)

    print("done loop")
    LEs = np.array([LEs[:ii].sum()/((ii+1)*dt) for ii in range(len(LEs))])
    print("done sum")
    return LEs[1:]

def point_mass():
    dt = 0.01
    nsteps = 10000

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

    t,q,p = integrator.run(x0.copy(), v0.copy(),
                           dt=dt, nsteps=nsteps)

    plt.clf()
    plt.plot(q[:,0,0], q[:,0,1], marker=None)
    plt.savefig("pt_mass_orbit.png")

    LE = lyapunov(q, p, dt, integrator)
    print("Lyapunov exponent computed")
    plt.clf()
    plt.plot(LE, marker=None)
    plt.savefig("pt_mass_le.png")

def pal5():
    dt = 0.1
    nsteps = 1000000

    from streams import usys
    X = 8.312877511
    Y = 0.242593717
    Z = 16.811943627
    x0 = np.array([[X,Y,Z]])

    Vx = (-52.429087*u.km/u.s).decompose(usys).value
    Vy = (-96.697363*u.km/u.s).decompose(usys).value
    Vz = (-8.156130*u.km/u.s).decompose(usys).value
    v0 = np.array([[Vx,Vy,Vz]])

    acc = np.zeros((1,3))
    potential = LawMajewski2010()
    integrator = LeapfrogIntegrator(potential._acceleration_at,
                                    func_args=(acc.shape[0],acc))

    t,q,p = integrator.run(x0.copy(), v0.copy(),
                           dt=dt, nsteps=nsteps)

    plt.clf()
    plt.figure(figsize=(10,10))
    plt.plot(np.sqrt(q[:,0,0]**2 + q[:,0,1]**2), q[:,0,2], marker=None)
    plt.xlim(0, 30)
    plt.ylim(-30, 30)
    plt.savefig('pal5_orbit.png')

    LE = lyapunov(q, p, dt, integrator)
    print("Lyapunov exponent computed")
    plt.clf()
    plt.semilogy(LE, marker=None)
    plt.savefig('pal5_le.png')

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

    dt = 0.1
    nsteps = 1000000

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

    else:
        return

    E = (600*100*(u.km/u.s)**2).decompose(usys).value
    V = zotos_potential(x0, *params)
    Lz = (10.*10.*u.km*u.kpc/u.s).decompose(usys).value # typo in paper? km/kpc instead of km*kpc

    vz = np.squeeze(np.sqrt(2*E - 2*V - Lz**2/x0[...,0]**2 - vx**2))
    v0 = np.array([[vx,0.,vz]])
    print(E, V, Lz**2/x0[...,0]**2, vx, vz)

    integrator = LeapfrogIntegrator(zotos_acceleration, func_args=params)
    t,q,p = integrator.run(x0.copy(), v0.copy(),
                           dt=dt, nsteps=nsteps)

    plt.clf()
    plt.figure(figsize=(10,10))
    plt.plot(np.sqrt(q[:,0,0]**2 + q[:,0,1]**2), q[:,0,2], marker=None)
    plt.xlim(0, 15)
    plt.ylim(-15, 15)
    plt.savefig('zotos_orbit.png')

    LE = lyapunov(q, p, dt, integrator)
    print("Lyapunov exponent computed")
    plt.clf()
    plt.plot(LE, marker=None)
    plt.savefig('zotos_le.png')

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

    else:
        return

    E = (600*100*(u.km/u.s)**2).decompose(usys).value
    V = zotos_potential(x0, *params)
    Lz = (10.*10.*u.km*u.kpc/u.s).decompose(usys).value # typo in paper? km/kpc instead of km*kpc

    vz = np.squeeze(np.sqrt(2*E - 2*V - Lz**2/x0[...,0]**2 - vx**2))
    v0 = np.array([[vx,0.,vz]])

    all_x0 = np.random.normal(x0, np.linalg.norm(x0)/25., size=(nstars,3))
    all_v0 = np.random.normal(v0, np.linalg.norm(v0)/25., size=(nstars,3))
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
    # point_mass()
    # pal5()
    zotos('4-3-box')
    zotos('chaos')
    # zotos_ball('4-3-box')
    # zotos_ball('2-1-box')
    # zotos_ball('8-5-box')
    # zotos_ball('3-2-box')
    # zotos_ball('chaos')

"""
ffmpeg -r 10 -i RZ_zotos_ball_%05d.png -codec h264 -r 10 -pix_fmt yuv420p 0RZ.mp4
ffmpeg -r 10 -i XZ_zotos_ball_%05d.png -codec h264 -r 10 -pix_fmt yuv420p 0XZ.mp4
"""