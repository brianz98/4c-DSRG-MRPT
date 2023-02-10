# Four-component DSRG-MRPT2
This is a pilot implementation of a four-component multi-reference many-body perturbation theory based on the DSRG-MRPT2 (**D**riven **S**imilarity **R**enormalization **G**roup second-order **M**ulti-**R**eference **P**erturbation **T**heory) formalism, based on a 4c-CASCI (four-component complete active space configuration interaction) reference.
We use an interface to PySCF to obtain the Dirac-Coulomb Hartree-Fock MOs and associated integrals, and extension to the Dirac-Coulomb-Breit or -Gaunt Hamiltonians would be trivial.
