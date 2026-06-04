# FRET Theory

## Förster Resonance Energy Transfer

FRET efficiency $E$ depends on the distance $r$ between a donor and an acceptor fluorophore according to the Förster equation:

$$E = \frac{1}{1 + (r/R_0)^6}$$

where $R_0$ is the Förster distance at which the transfer efficiency is 50%.

## Ensemble Averaging

In solution-state biophysics, we often observe an ensemble average of efficiencies:

$$\langle E \rangle = \frac{1}{N} \sum_{i=1}^N E(r_i)$$

**diff-fret** uses JAX's `vmap` to efficiently compute these averages over large conformational trajectories.
