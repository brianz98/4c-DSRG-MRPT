# Four-component DSRG-MRPT2/3
This is a pilot implementation of a four-component multi-reference many-body perturbation theory based on the DSRG-MRPT2 (**D**riven **S**imilarity **R**enormalization **G**roup second-order **M**ulti-**R**eference **P**erturbation **T**heory) formalism, based on a 4c-CASCI or 4c-CASSCF reference.

## Current capabilities
- Interface to PySCF, specifically for the Dirac-Hartree-Fock solver and its interface to libcint for integrals. The DC integrals are density fitted, the DCG and DCB cannot be currently density fitted.
- NR/4C-DHF integral transformation
  - Ability to read in MO coefficients from other sources, specifically [PySCF](https://pyscf.org/index.html), [Psi4](https://psicode.org/psi4manual/master/index.html) and [ChronusQ](https://github.com/xsligroup/chronusq_public/wiki), the latter enables using 4C-CASSCF references.
  - Density fitting is available throughout the code
  - Frozen core / deleted virtual approximations
- NR/4C-MP2
- NR/4C-CASCI solver
- NR/4C-DSRG-MRPT2/3, based on either (NR/4C-)CASCI or (NR/4C-)CASSCF references
  - Use of semicanonical orbitals
  - Full reference relaxation
  - State-averaging
- NR/4C Adaptive CI (ACI)
  - TD extension (TD-ACI) for ionized states.

## To-dos
- [ ] Memory optimization
- [ ] State-averaged ACI
