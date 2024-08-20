"""
Constants for LBM. There are too many...
"""
import jax.numpy as jnp

# set up space
N_POINTS_X, N_POINTS_Y = 300, 50
BODY_CENTER_IDX_X = N_POINTS_X // 5
BODY_CENTER_IDX_Y = N_POINTS_Y // 2
BODY_RADIUS_IDX = N_POINTS_Y // 9

# additional properties and flags
N = 15000
REYNOLDS_NUM = 80
MAX_INFLOW_VELOCITY = 0.04
VIZUALIZE = True
PLOT_EVERY_N_STEPS = 100
SKIP_FIRST_N_ITER = 0

# set up D2Q9 lbm grid
N_DISCRETE_VELOCITIES = 9

# 09 unit velocities and relevant indices
LATTICE_VELOCITIES = jnp.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                                [0, 0, 1, 0, -1, 1, 1, -1, -1]])
LATTICE_IDX = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
OPP_LATTICE_IDX = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
RIGHT_VELOCITIES = jnp.array([1, 5, 8])
LEFT_VELOCITIES = jnp.array([3, 6, 7])
UP_VELOCITIES = jnp.array([2, 5, 6])
DOWN_VELOCITIES = jnp.array([4, 7, 8])
PURE_VERTICAL_VELOCITIES = jnp.array([0, 2, 4])
PURE_HORIZONTAL_VELOCITIES = jnp.array([0, 1, 3])

# related to navier stokes (out of scope :p)
LATTICE_WEIGHTS = jnp.array([
    4/9,                    # center
    1/9, 1/9/ 1/9, 1/9,     # axis-aligned
    1/36, 1/36, 1/36, 1/36  # 45 deg.
])
