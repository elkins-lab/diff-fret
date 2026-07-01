import typing

import jax
import jax.numpy as jnp

from diff_fret.kernels import fret_efficiency_av


def test_fret_av_differentiable() -> None:
    """
    Verify that we can take gradients through the Accessible Volume (AV) simulation.
    """
    attachment_d = jnp.array([0.0, 0.0, 0.0])
    attachment_a = jnp.array([50.0, 0.0, 0.0])

    def loss(pos_d: jnp.ndarray) -> typing.Any:
        return fret_efficiency_av(pos_d, attachment_a, key=jax.random.PRNGKey(0), n_samples=10)  # type: ignore[no-any-return]

    grads = jax.grad(loss)(attachment_d)
    assert grads.shape == attachment_d.shape
    assert not jnp.any(jnp.isnan(grads))


def test_fret_av_differentiable_radii() -> None:
    """
    Verify that we can take gradients with respect to dye radii.
    """
    attachment_d = jnp.array([0.0, 0.0, 0.0])
    attachment_a = jnp.array([50.0, 0.0, 0.0])

    def loss(radius_d: float) -> typing.Any:
        return fret_efficiency_av(
            attachment_d,
            attachment_a,
            key=jax.random.PRNGKey(0),
            radius_donor=radius_d,
            radius_acceptor=10.0,
            n_samples=10,
        )  # type: ignore[no-any-return]

    grad_val = jax.grad(loss)(10.0)
    assert not jnp.isnan(grad_val)


def test_fret_av_vs_point() -> None:
    """
    Verify that AV averaging behaves reasonably compared to point-to-point.
    """
    pos_d = jnp.array([0.0, 0.0, 0.0])
    pos_a = jnp.array([50.0, 0.0, 0.0])

    # Point-to-point efficiency at 50A with R0=50A is 0.5
    # With a small radius, AV should be close to 0.5
    eff_av = fret_efficiency_av(
        pos_d,
        pos_a,
        key=jax.random.PRNGKey(0),
        radius_donor=1.0,
        radius_acceptor=1.0,
        n_samples=100,
    )
    assert jnp.allclose(eff_av, 0.5, atol=0.05)


def test_distance_distribution_identical_coords() -> None:
    """
    Test distance distribution and its gradient with identical coords (r=0).
    The gradient shouldn't be NaN because of the safe masking.
    """
    pos_d = jnp.array([10.0, 10.0, 10.0])
    pos_a = jnp.array([10.0, 10.0, 10.0])

    from diff_fret.kernels import distance_distribution

    # Distance should be zero
    dist = distance_distribution(pos_d, pos_a)
    assert jnp.allclose(dist, 0.0)

    # Gradient should not be NaN
    def loss(p: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(distance_distribution(p, pos_a))

    grad = jax.grad(loss)(pos_d)
    assert not jnp.any(jnp.isnan(grad))
