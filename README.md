# Updating-VACF
A velocity autocorrelation function on an updating atom group with options for decomposition with respect to an axis

## About
The main code, updating_vacf.py, calculates an autocorrelation function on a subset of atoms whose identities can change throughout the trajectory (e.g. water oxygen atoms within 5 Angstroms of a chosen residue. Only atoms present at $t$ and $t+\tau$ contribute to the function. Additionally, the code can decompose the ACF with respect to a chosen axis. Originally, this code was written to study the diffusion of water around an amyloid fibril. In the future, I hope to make this code more versatile, but for now you will likely have to modify it (at least prepare_vacf.py) for your given system.
