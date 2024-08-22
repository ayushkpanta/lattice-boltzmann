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
    VISUALIZE, PLOT_EVERY_N_STEPS, SKIP_FIRST_N_ITER,
    N_DISCRETE_VELOCITIES, LATTICE_VELOCITIES, LATTICE_IDX,
    OPP_LATTICE_IDX, RIGHT_VELOCITIES, LEFT_VELOCITIES,
    UP_VELOCITIES, DOWN_VELOCITIES, PURE_VERTICAL_VELOCITIES,
    PURE_HORIZONTAL_VELOCITIES, LATTICE_WEIGHTS
)

def main():
    # double precision enable
    jax.config.update("jax_enable_x64", True)

    kinematic_vicsosity = (MAX_INFLOW_VELOCITY * BODY_RADIUS_IDX) / REYNOLDS_NUM
    relaxation_omega = 1.0 / (3.0 * kinematic_vicsosity + 0.5)

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

    @jax.jit
    def update_sim(prev_discrete_velocities):

        # 1. looking at the outflow region -> stopping it from flowing back in
        prev_discrete_velocities = prev_discrete_velocities.at[-1, :, LEFT_VELOCITIES].set(prev_discrete_velocities[-2, :, LEFT_VELOCITIES])

        # 2. macroscopic vel 
        prev_density = get_density(prev_discrete_velocities)
        prev_macroscopic_velocities = get_macroscopic_velocities(
            prev_discrete_velocities, prev_density)
        
        # 3. inflow dirichlet with zou/he scheme (technical)
        # essentially exluding top and bottom boundaries
        prev_macroscopic_velocities = prev_macroscopic_velocities.at[0, 1:-1, :].set(velocity_profile[0, 1:-1, :])
        prev_density = prev_density.at[0, :].set(
            (
                get_density(prev_discrete_velocities[0, :, PURE_VERTICAL_VELOCITIES].T)
                + 
                2 * 
                get_density(prev_discrete_velocities[0, :, LEFT_VELOCITIES].T)
            ) / (
                1 - prev_macroscopic_velocities[0, :, 0]
            )
        )

        # 4. compute discrete equilibria velocities
        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
            prev_macroscopic_velocities, prev_density
        )

        # 3.2. more of zou/he scheme
        prev_discrete_velocities = prev_discrete_velocities.at[0, :, RIGHT_VELOCITIES].set(
            equilibrium_discrete_velocities[0, :, RIGHT_VELOCITIES]
        )

        # 5. LB Collision according to BGK
        discrete_velocities_post_collision = (
            prev_discrete_velocities - relaxation_omega *
            (prev_discrete_velocities - equilibrium_discrete_velocities)
        )

        # (6) bounce-back for no-slip conditions
        # vel flipped if belong to object mask to negate the velocity
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_collision = discrete_velocities_post_collision.at[obstacle_mask, LATTICE_IDX[i]].set(prev_discrete_velocities[obstacle_mask, OPP_LATTICE_IDX[i]])

        # 7. Stream alongside lattice velocities
        # periodic boundary velocities, last piece of the puzzle
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:,:, i].set(
                # 
                jnp.roll(
                    jnp.roll(
                        discrete_velocities_post_collision[:, :, i],
                        LATTICE_VELOCITIES[0, i],
                        axis = 0,
                    ),
                    LATTICE_VELOCITIES[1, i],
                    axis = 1
                )
            )

        return discrete_velocities_streamed
    
    # create initial velocity profile

    # macro to micro discrete velocities
    # 1 density all around

    discrete_velocities_prev = get_equilibrium_discrete_velocities(
        velocity_profile, jnp.ones((N_POINTS_X, N_POINTS_Y)),
    )

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6), dpi=100)

    for iter_idx in tqdm(range(N)):
        discrete_velocities_nxt = update_sim(discrete_velocities_prev)
        discrete_velocities_prev = discrete_velocities_nxt

        # viz appropriate
        if iter_idx % PLOT_EVERY_N_STEPS == 0 and VISUALIZE and iter_idx > SKIP_FIRST_N_ITER:

            density = get_density(discrete_velocities_nxt)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_nxt,
                density,
            )
            # getting curl for lower graph...
            velocity_magnitude = jnp.linalg.norm(
                macroscopic_velocities, axis = -1, ord = 2
            )
            dUdX, dUdY = jnp.gradient(macroscopic_velocities[..., 0])
            dVdX, dVdY = jnp.gradient(macroscopic_velocities[..., 1])
            curl = (dUdY - dVdX)

            # velocity magnitude contour plot
            plt.subplot(211)
            plt.contourf(
                X, Y, velocity_magnitude, levels = 50, cmap = cmr.amber
            )
            plt.colorbar().set_label("Velocity Magnitude")
            plt.gca().add_patch(
                plt.Circle(
                    (BODY_CENTER_IDX_X, BODY_CENTER_IDX_Y), BODY_RADIUS_IDX, color = 'grey',
                )
            )

            # vorticity magnitude contour plot
            plt.subplot(212)
            plt.contourf(
                X, Y, curl, levels = 50, cmap = cmr.redshift, vmin = -0.02, vmax = 0.02
            )
            plt.colorbar().set_label("Vorticity Magnitude")
            plt.gca().add_patch(
                plt.Circle(
                    (BODY_CENTER_IDX_X, BODY_CENTER_IDX_Y), BODY_RADIUS_IDX, color = 'grey',
                )
            )

            plt.draw()
            plt.pause(0.0001)
            plt.clf()

    if VISUALIZE:
        plt.show()



if __name__ == "__main__":
    main()
