"""
Functions for computation.
"""
import jax.numpy as jnp
from constants import (
    N_POINTS_X, N_POINTS_Y, BODY_CENTER_IDX_X, BODY_CENTER_IDX_Y,
    BODY_RADIUS_IDX, N, REYNOLDS_NUM, MAX_INFLOW_VELOCITY,
    VISUALIZE, PLOT_EVERY_N_STEPS, SKIP_FIRST_N_ITER,
    N_DISCRETE_VELOCITIES, LATTICE_VELOCITIES, LATTICE_IDX,
    OPP_LATTICE_IDX, RIGHT_VELOCITIES, LEFT_VELOCITIES,
    UP_VELOCITIES, DOWN_VELOCITIES, PURE_VERTICAL_VELOCITIES,
    PURE_HORIZONTAL_VELOCITIES, LATTICE_WEIGHTS
)

def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis = -1)
    return density

# REMEMBER: D2Q9 standard
def get_macroscopic_velocities(discrete_velocities, density):
    # einsum key:
        # discrete dim: N * M * Q (points x * points y * num discrete vel.)
        # lattice dim: d * Q (macro * microscopic velocities)
        # output: N * M * d (points x * points y * macro dim.)
    summation = jnp.einsum("NMQ,dQ->NMd", discrete_velocities, LATTICE_VELOCITIES)
    # must add dummy axis for dimensionality
    macroscopic_velocities = summation / density[..., jnp.newaxis]
    return macroscopic_velocities

def get_equilibrium_discrete_velocities(macroscopic_velocities, density):

    projected_discrete_velocities = jnp.einsum("dQ,NMd->NMQ",LATTICE_VELOCITIES, macroscopic_velocities)

    macro_velocity_mag = jnp.linalg.norm(macroscopic_velocities, axis = -1, ord=2)

    equilibrium_discrete_velocities = (
        density[..., jnp.newaxis] * 
        LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :] * 
        (1
         +
         (3 * projected_discrete_velocities)
         +
         (9/2 * projected_discrete_velocities**2)
         - 
         (3/2 * macro_velocity_mag[..., jnp.newaxis]**2)
        )
    )

    return equilibrium_discrete_velocities
