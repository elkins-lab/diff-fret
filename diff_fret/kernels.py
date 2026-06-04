import jax.numpy as jnp


def distance_distribution(
    donor_coords: jnp.ndarray,
    acceptor_coords: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute donor-acceptor distances.

    Args:
        donor_coords: (N, 3) or (3,) coordinates of donor(s).
        acceptor_coords: (N, 3) or (3,) coordinates of acceptor(s).

    Returns:
        Distances (N,).
    """
    dist_sq = jnp.sum((donor_coords - acceptor_coords) ** 2, axis=-1)
    # Safe distance for gradients (avoids NaN at dist=0)
    dist = jnp.sqrt(jnp.where(dist_sq > 0, dist_sq, 1.0))
    return jnp.where(dist_sq > 0, dist, 0.0)


def fret_efficiency(
    r: jnp.ndarray,
    r0: float = 50.0,
) -> jnp.ndarray:
    """
    Compute FRET efficiency using Förster theory.

    Args:
        r: Distance(s) in Angstroms.
        r0: Förster distance in Angstroms (default 50.0).

    Returns:
        FRET efficiency E in [0, 1].
    """
    return 1.0 / (1.0 + (r / r0) ** 6)


def average_efficiency(
    coords_donor: jnp.ndarray,
    coords_acceptor: jnp.ndarray,
    r0: float = 50.0,
) -> jnp.ndarray:
    """
    Compute ensemble-averaged FRET efficiency.

    Args:
        coords_donor: (M, 3) donor coordinates for M frames.
        coords_acceptor: (M, 3) acceptor coordinates for M frames.
        r0: Förster distance.

    Returns:
        Scalar averaged efficiency <E>.
    """
    r = distance_distribution(coords_donor, coords_acceptor)
    e = fret_efficiency(r, r0)
    return jnp.mean(e)
