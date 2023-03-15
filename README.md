# Four-component DSRG-MRPT2
This is a pilot implementation of a four-component multi-reference many-body perturbation theory based on the DSRG-MRPT2 (**D**riven **S**imilarity **R**enormalization **G**roup second-order **M**ulti-**R**eference **P**erturbation **T**heory) formalism, based on a 4c-CASCI (four-component complete active space configuration interaction) reference.

## Current capabilities
- Interface to PySCF, specifically for the Dirac-Hartree-Fock solver and its interface to libcint for integrals.
- NR/4C-DHF integral transformation
  - Ability to read in MO coefficients from other sources, specifically [Psi4](https://psicode.org/psi4manual/master/index.html) and [ChronusQ](https://urania.chem.washington.edu/chronusq/chronusq_public/-/wikis/home), the latter enables using 4C-CASSCF references.
  - Density fitting is available throughout the code
  - Frozen core / deleted virtual approximations
- NR/4C-MP2
- NR/4C-CASCI solver
- NR/4C-DSRG-MRPT2, based on either (NR/4C-)CASCI or (NR/4C-)CASSCF references
  - Use of semicanonical orbitals
  - Partial reference relaxation

## To-dos
- [ ] Full reference relaxation
- [ ] Memory optimization
- [ ] DSRG-MRPT3
