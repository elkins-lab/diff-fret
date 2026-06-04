# 📏 diff-fret

**diff-fret** provides high-performance, auto-differentiable kernels for modeling Fluorescence Resonance Energy Transfer (FRET) observables from structural ensembles.

## Quick Start

```python
import jax.numpy as jnp
from diff_fret.kernels import fret_efficiency

# Distance in Angstroms
r = jnp.array([45.0, 50.0, 55.0])
# Compute efficiency with R0 = 50.0
e = fret_efficiency(r, r0=50.0)
print(e)
```
