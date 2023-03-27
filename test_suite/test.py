import os
import sys
sys.path.append(os.path.relpath("../"))
from relforte import RelForte
import pyscf
import numpy as np

RTOL=0.0

mol = pyscf.gto.M(
    verbose = 2,
    atom = '''
H 0 0 0
F 0 1.5 0
''',
    basis = 'cc-pvdz', spin=0, charge=0, symmetry=False
)
nonrel = RelForte(mol, verbose=True, density_fitting=False, decontract=False)
nonrel.run_rhf(transform=True, debug=True, frozen=(0,0), dump_mo_coeff=False)
nonrel.run_mp2(relativistic=False)
nonrel.run_casci(cas=(6,8), do_fci=False, rdm_level=3, relativistic=False, semi_canonicalize=True)
nonrel.run_dsrg_mrpt2(s=2.0, relativistic=False, relax='iterate')

rel = RelForte(mol, verbose=False, density_fitting=False, decontract=False)
rel.run_dhf(transform=True, debug=True, frozen=(0,0), dump_mo_coeff=False)
rel.run_mp2(relativistic=True)
rel.run_casci(cas=(6,8), do_fci=False, rdm_level=3, relativistic=True, semi_canonicalize=True)
rel.run_dsrg_mrpt2(s=2.0, relativistic=True, relax=None)

def test_rhf_rebuild():
    assert np.isclose(nonrel.rhf_e_rebuilt, -99.8728524746859, rtol=RTOL)

def test_nonrel_mp2():
    assert np.isclose(nonrel.e_mp2, -0.2281881882243305, rtol=RTOL)

def test_nonrel_casci():
    assert np.isclose(nonrel.e_casci_save, -99.91039280165454, rtol=RTOL)

def test_nonrel_casci_rebuild():
    assert np.isclose(nonrel.e_casci_rebuilt, -99.91039280165454, rtol=RTOL)

def test_nonrel_casci_pyscf():
    assert np.isclose(nonrel.e_casci_rebuilt, nonrel.e_casci_pyscf, rtol=RTOL)