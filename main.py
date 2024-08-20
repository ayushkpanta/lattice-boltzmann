"""
Lattice-Boltzmann Method (LBM) for CNT Simulations.

About LBM: https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods
Inspiration: https://www.youtube.com/watch?v=ZUXmO4hu-20&t=947s

Why? LBM combines microscopic and macroscopic behaviors that are useful for computational fluid dynamics simulations. It does so by creating a state space, encoding for position and velocity. In 2D, state space is discretized by considering 9 points, representative of unit changes in velocity.

In this software, I will start by following the YouTube video to create an LBM simulation for water passing through a porous membrane, which is to serve as proxy for a CNT forest of a certain density. Then, I will attempt to move on and implement it in 3D to capture realistic fluid movement.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm
from computations import (
    get_density, get_macroscopic_velocities, get_equilibrium_discrete_velocities)
from constants import (
    N_POINTS_X, N_POINTS_Y, BODY_CENTER_IDX_X, BODY_CENTER_IDX_Y,
    BODY_RADIUS_IDX, N, REYNOLDS_NUM, MAX_INFLOW_VELOCITY,
    VIZUALIZE, PLOT_EVERY_N_STEPS, SKIP_FIRST_N_ITER,
    N_DISCRETE_VELOCITIES, LATTICE_VELOCITIES, LATTICE_IDX,
    OPP_LATTICE_IDX, RIGHT_VELOCITIES, LEFT_VELOCITIES,
    UP_VELOCITIES, DOWN_VELOCITIES, PURE_VERTICAL_VELOCITIES,
    PURE_HORIZONTAL_VELOCITIES, LATTICE_WEIGHTS
)

def main():
    # double precision enable
    jax.config.update("jax_enable_x64", True)

    kinematic_vicosity = (MAX_INFLOW_VELOCITY * BODY_RADIUS_IDX) / REYNOLDS_NUM
    relaxation_omega = 1.0 / (3.0 * kinematic_vicosity + 0.5)

    # mesh creation
    x, y = jnp.arange(N_POINTS_X), jnp.arange(N_POINTS_Y)
    X, Y = jnp.meshgrid(x, y, indexing = "ij")

    # obstacle mask (body) ->>> THIS NEEDS TO BE CHANGED!
    # array like X,Y where True if overlap with object, False otherwise
    obstacle_flag_matrix = jnp.sqrt(
        (X - BODY_CENTER_IDX_X)**2
        + (Y - BODY_CENTER_IDX_Y)**2)
    obstacle_mask = obstacle_flag_matrix < BODY_RADIUS_IDX

    # we index to set velocity
    # THIS WILL BE OUR PROXY FOR THE PRESSURE FROM PUMP
    # not mutable arrays in jax?!?!
    velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    # velocity_profile[:, :, 0] = MAX_INFLOW_VELOCITY # numpy style
    velocity_profile = velocity_profile.at[:, :, 0].set(MAX_INFLOW_VELOCITY)


if __name__ == "__main__":
    main()
