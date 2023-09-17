import pyscf
from pyscf import mp, mcscf
import numpy as np
import h5py
import time
import scipy
import itertools
import copy
import os, glob
import warnings
from contractions import *
from wicked_contractions import *
import gc

MACHEPS = 1e-9
EH_TO_WN = 219474.63
EH_TO_EV = 27.211399

def zero_mat(mat):
    mat[np.abs(mat) < 1e-12] = 0
    return mat

def antisymmetrize_and_hermitize(T):
    # antisymmetrize the residual
    T += np.einsum("ijab->abij",T.conj()) # This is the Hermitized version (i.e., [H,A]), which should then be antisymmetrized
    temp = T.copy()
    T -= np.einsum("ijab->jiab", temp)
    T += np.einsum("ijab->jiba", temp)
    T -= np.einsum("ijab->ijba", temp)

def compute_ht_h1_only(fixed_args, C1, C2, H1, T1, T2):
    H1_T2_C2(None, C2, H1, None, None, T2, *fixed_args)
    H1_T2_C1(C1, None, H1, None, None, T2, *fixed_args)
    H1_T1_C1(C1, None, H1, None, T1, None, *fixed_args)
    antisymmetrize_and_hermitize(C2)
    C1 = C1 + C1.T.conj()

def compute_ht(fixed_args, C1, C2, H1, H2, T1, T2):
    H1_T1_C1(C1, None, H1, None, T1, None, *fixed_args)
    H1_T2_C1(C1, None, H1, None, None, T2, *fixed_args)
    H2_T1_C1(C1, None, None, H2, T1, None, *fixed_args)
    H2_T2_C1(C1, None, None, H2, None, T2, *fixed_args)
    H1_T2_C2(None, C2, H1, None, None, T2, *fixed_args)
    H2_T1_C2(None, C2, None, H2, T1, None, *fixed_args)
    H2_T2_C2(None, C2, None, H2, None, T2, *fixed_args)
    antisymmetrize_and_hermitize(C2)
    C1 = C1 + C1.T.conj()

def compute_ht_c0(fixed_args, H1, H2, T1, T2):
    C0 = H_T_C0(None, None, H1, H2, T1, T2, *fixed_args)
    return C0

def regularized_denominator(x, s):
    z = np.sqrt(s) * x
    if abs(z) <= MACHEPS:
        return np.sqrt(s)*(z - z**3/2 + z**5/6)
    return (1. - np.exp(-s * x**2)) / x

def set_bit(bit_loc):
    """
    Set the bit_loc-th bit in bit string f. Returns unchanged if the bit is already set. bit_loc is zero-indexed.
    """
    f = 0
    for loc in bit_loc:
        f = f | 1<<loc
    return f

def set_bit_single(f, bit_loc):
    """
    Set the bit_loc-th bit in bit string f. Returns unchanged if the bit is already set. bit_loc is zero-indexed.
    """
    return f | 1<<bit_loc

def clear_bit(f, bit_loc):
    """
    Unset the bit_loc-th bit in bit string f. Returns unchanged if the bit is already unset. bit_loc is zero-indexed.
    """
    return f & ~(1<<bit_loc)

def test_bit(f, bit_loc):
    """
    Test if bit_loc in f is set. Returns 1 if set, 0 if not set.
    """
    return (f & (1<<bit_loc)) >> bit_loc

def count_set_bits(f):
    """
    Return the number of set (1) bits in the bit string f.
    """
    return int(bin(f).count('1'))

def get_excitation_level(f1, f2):
    """
    Get the excitation level between two bit strings f1 and f2, i.e., half the Hamming distance.
    """
    return int(count_set_bits(f1^f2)/2)

def annop(bit_string, ispinor):
    """
    Annihilation operator, annihilates spinorb in bit_string, returns the sign and the resulting bit string.
    If spinorb is already empty, sign is zero and the bit string is returned unchanged.
    """
    if (not test_bit(bit_string, ispinor)):
        sgn = 0
    else:
        test_string = 0
        for i in range(ispinor):
            test_string = set_bit_single(test_string, i)
        sgn = (-1)**(count_set_bits(bit_string & test_string))
        bit_string = clear_bit(bit_string, ispinor)
    return (sgn, bit_string)
    
def bstring_to_occ_vec(f, nelec, norbs):
    occ_vec = np.zeros(nelec, dtype='int')
    nfound = 0
    for i in range(norbs):
        if test_bit(f, i)==1:
            occ_vec[nfound] = i
            nfound += 1
            if (nfound==nelec):
                break
    return occ_vec

def bstring_to_unocc_vec(f, nelec, norbs):
    unocc_vec = np.zeros(norbs-nelec, dtype='int')
    nfound = 0
    for i in range(norbs):
        if test_bit(f, i)==0:
            unocc_vec[nfound] = i
            nfound += 1
            if (nfound==norbs-nelec):
                break
    return unocc_vec

def get_excit_connection(f1, f2, exlvl, nelec, norbs):
    excit_bstring = f1^f2
    
    excit = np.zeros((2,exlvl), dtype='int')
    nbit_f1_found = 0
    nbit_f2_found = 0
    for i in range(norbs):
        if (test_bit(excit_bstring, i)==1):
            # Check where this electron is coming from / going to
            if (test_bit(f1, i)):
                excit[0][nbit_f1_found] = i
                nbit_f1_found += 1
            else:
                excit[1][nbit_f2_found] = i
                nbit_f2_found += 1
            if (nbit_f1_found == exlvl and nbit_f2_found==exlvl):
                break
                
    # Get permutation!
    perm = annop_mult(f1, excit[0])[0] * annop_mult(f2, excit[1])[0]

    return excit, perm

def annop_mult(f, orbs):
    fold = f
    perm = 1
    for orb in orbs:
        iperm, fnew = annop(fold, orb)
        perm *= iperm
        fold = fnew
    
    return perm, fnew

def make_cumulants(rdm):
    try:
        assert rdm['max_rdm_level'] >= 2
    except AssertionError:
        raise Exception('Max RDM level is 1, cumulants not necessary!')
        
    _lamb = {'max_cumulant_level':rdm['max_rdm_level']}
    _lamb['gamma1'] = rdm['1rdm']
    _lamb['eta1'] = -rdm['1rdm'] + np.diag(np.zeros(rdm['1rdm'].shape[0])+1)
    _lamb['lambda2'] = rdm['2rdm'] - np.einsum('pr,qs->pqrs', rdm['1rdm'], rdm['1rdm']) + np.einsum('ps,qr->pqrs', rdm['1rdm'], rdm['1rdm'])
    if (rdm['max_rdm_level'] == 3):
        _lamb['lambda3'] = rdm['3rdm'] - np.einsum('ps,qrtu->pqrstu',rdm['1rdm'],rdm['2rdm']) + np.einsum('pt,qrsu->pqrstu',rdm['1rdm'],rdm['2rdm']) + np.einsum('pu,qrts->pqrstu',rdm['1rdm'],rdm['2rdm'])- np.einsum('qt,prsu->pqrstu',rdm['1rdm'],rdm['2rdm']) + np.einsum('qs,prtu->pqrstu',rdm['1rdm'],rdm['2rdm']) + np.einsum('qu,prst->pqrstu',rdm['1rdm'],rdm['2rdm'])- np.einsum('ru,pqst->pqrstu',rdm['1rdm'],rdm['2rdm']) + np.einsum('rs,pqut->pqrstu',rdm['1rdm'],rdm['2rdm']) + np.einsum('rt,pqsu->pqrstu',rdm['1rdm'],rdm['2rdm'])+ 2*(np.einsum('ps,qt,ru->pqrstu',rdm['1rdm'],rdm['1rdm'],rdm['1rdm']) + np.einsum('pt,qu,rs->pqrstu',rdm['1rdm'],rdm['1rdm'],rdm['1rdm']) + np.einsum('pu,qs,rt->pqrstu',rdm['1rdm'],rdm['1rdm'],rdm['1rdm']))- 2*(np.einsum('ps,qu,rt->pqrstu',rdm['1rdm'],rdm['1rdm'],rdm['1rdm']) + np.einsum('pu,qt,rs->pqrstu',rdm['1rdm'],rdm['1rdm'],rdm['1rdm']) + np.einsum('pt,qs,ru->pqrstu',rdm['1rdm'],rdm['1rdm'],rdm['1rdm']))
                
    return _lamb

def cq_to_pyscf(mol, bin_name):
    """
    Reorder ChronusQuantum DHF MO coeffs to PySCF order, and transform to spinor AO basis.
    CQ order:    [La1, La2, ..., Sa1, Sa2, ..., Lb1, Lb2, ..., Sb1, Sb2, ...]
    PySCF order: [La1, Lb1, ..., Sa1, Sb1, ...]
    """
    f = h5py.File(bin_name, 'r')
    mo_cq = f['SCF']['MO1'][:]

    nspinor = mo_cq.shape[0]//2
    coeffs_reorder = np.zeros_like(mo_cq)
    coeffs_reorder[::2,:] = mo_cq[:,:nspinor].T
    coeffs_reorder[1::2,:] = mo_cq[:,nspinor:].T

    coeffs_spinor = sph_to_spinor(mol, coeffs_reorder)[0]

    return coeffs_spinor

def sph_to_spinor(mol, coeffs):
    """
    Transfer DHF MOs coefficients in a real spherical AO spinor basis,
    >>> e.g. {p_xa, p_xb, p_ya, p_yb, p_za, p_zb},
    >>> each of the above is a two-spinor, for example, p_xa=(px, 0), p_xb=(0, px) etc.
    to the corresponding complex spinor basis.
    >>> e.g. {p_{1/2,-1/2}, p_{1/2,+1/2}, p_{3/2,-3/2}, p_{3/2,-1/2}, p_{3/2,+1/2}, p_{3/2,+3/2}},
    >>> each of the above is a two-spinor, for example, p_{1/2,-1/2} = (x-iy, z)/sqrt(3)
    """
    nspinor = coeffs.shape[0]//2
    rotmat = np.zeros((nspinor,nspinor), dtype='complex128')
    rotmat[::2,:] = mol.sph2spinor_coeff()[0]
    rotmat[1::2,:] = mol.sph2spinor_coeff()[1]

    mo_rot = coeffs.copy()
    # To see why the complex conjugate needs to be taken, we use an example:
    # p_{1/2,-1/2} \propto (x-iy, z), so our rotmat gives for the column of p_{1/2,-1/2}
    # [1, 0, -i, 0, 0, -1] for the real spherical AO spinors [pxa, pxb, pya, pyb, pza, pzb].
    # This means to rotate this linear combination of real spherical p spinors to p_{1/2,-1/2} 
    # (i.e., [-1, 0, 0, 0, 0, 0] in the complex representation),
    # we need to dot it with [1, 0, i, 0, 0, -1], which suggests that we need to take the complex conjugate of the rotmat.
    mo_rot[:nspinor,:] = np.einsum('uv,ui->vi', np.conj(rotmat), coeffs[:nspinor,:])
    mo_rot[nspinor:,:] = np.einsum('uv,ui->vi', np.conj(rotmat), coeffs[nspinor:,:])
    
    return mo_rot, rotmat

def form_cas_hamiltonian(H1body, H2body, det_strings, verbose, cas, ncore=0):
    ncombs = len(det_strings)
    _mem = ncombs**2*16/1e9
    if (_mem < 1.0):
        if (verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the CASCI Hamiltonian!')
    else:
        if (verbose): print(f'Will now allocate {_mem:.3f} GB memory for the CASCI Hamiltonian!')
    _hamil_det = np.zeros((ncombs,ncombs), dtype='complex128')
    for i in range(ncombs):
        for j in range(i+1):
            exlvl = get_excitation_level(det_strings[i], det_strings[j])
            if (exlvl <= 2):
                if (i==j):
                    occ = bstring_to_occ_vec(det_strings[i], *cas)
                    _hamil_det[i,i] = 0
                    for iocc in occ:
                        _hamil_det[i,i] += H1body[iocc,iocc]
                        for jocc in occ:
                            _hamil_det[i,i] += 0.5*H2body[iocc+ncore,jocc+ncore,iocc+ncore,jocc+ncore]
                elif (exlvl == 1):
                    occ = bstring_to_occ_vec(det_strings[i], *cas)
                    conn, perm = get_excit_connection(det_strings[i], det_strings[j], exlvl, *cas)
                    _hamil_det[i,j] = H1body[conn[0],conn[1]]
                    for iocc in occ:
                        _hamil_det[i,j] += H2body[conn[0]+ncore, iocc+ncore, conn[1]+ncore, iocc+ncore]

                    _hamil_det[i,j] *= perm
                    _hamil_det[j,i] = np.conj(_hamil_det[i,j])
                elif (exlvl == 2):
                    conn, perm = get_excit_connection(det_strings[i], det_strings[j], exlvl, *cas)
                    _hamil_det[i,j] = perm*H2body[conn[0][0]+ncore, conn[0][1]+ncore, conn[1][0]+ncore, conn[1][1]+ncore]
                    _hamil_det[j,i] = np.conj(_hamil_det[i,j])         
    return _hamil_det

def get_hamil_element(f0, f1, H1body, H2body, cas, ncore=0):
    """
    <f0|H|f1>
    """
    _hmatel = 0.0j
    exlvl = get_excitation_level(f0, f1)
    if (exlvl <= 2):
        if (exlvl == 0):
            occ = bstring_to_occ_vec(f0, *cas)
            _hmatel = 0
            for iocc in occ:
                _hmatel += H1body[iocc,iocc]
                for jocc in occ:
                    _hmatel += 0.5*H2body[iocc+ncore,jocc+ncore,iocc+ncore,jocc+ncore]
        elif (exlvl == 1):
            occ = bstring_to_occ_vec(f0, *cas)
            conn, perm = get_excit_connection(f0, f1, exlvl, *cas)
            _hmatel = H1body[conn[0],conn[1]]
            for iocc in occ:
                _hmatel += H2body[conn[0]+ncore, iocc+ncore, conn[1]+ncore, iocc+ncore]
            _hmatel *= perm
        elif (exlvl == 2):
            conn, perm = get_excit_connection(f0, f1, exlvl, *cas)
            _hmatel = perm*H2body[conn[0][0]+ncore, conn[0][1]+ncore, conn[1][0]+ncore, conn[1][1]+ncore]
    
    return _hmatel

def form_cas_determinant_strings(nelec, norbs):
    """
    Returns the unrestricted (no spin or spatial symmetry imposed) list of bitstrings in a (nelec, n(spin(or))orbs) CAS.
    """
    ncombs = scipy.special.comb(norbs, nelec, exact=True)
    det_strings = list(map(set_bit, list(itertools.combinations(range(norbs),nelec))))
    assert(len(det_strings) == ncombs)

    return ncombs, det_strings

def form_cas_determinant_strings_general(occlist, actlist, nelec):
    """
    Returns the unrestricted (no spin or spatial symmetry imposed) list of bitstrings in the active space generated by distributing nelec in the occlist.
    """
    # just in case..
    actlist = list(np.sort(actlist)) 
    occlist = list(np.sort(occlist)) 
    norbs = len(actlist)
    
    ncombs = scipy.special.comb(norbs, nelec, exact=True)
    det_strings = list(map(set_bit, [list(_) + occlist for _ in list(itertools.combinations(actlist,nelec))] ))
    assert(len(det_strings) == ncombs)

    return ncombs, det_strings

def get_1_rdm_sa(det_strings, cas, states, weights, verbose):
    _sa_rdm = np.zeros((cas[1],cas[1]), dtype='complex128')
    for i in range(len(weights)):
        _sa_rdm += weights[i] * get_1_rdm(det_strings, cas, states[i], verbose)
    return _sa_rdm

def get_2_rdm_sa(det_strings, cas, states, weights, verbose):
    _sa_rdm = np.zeros((cas[1],cas[1],cas[1],cas[1]), dtype='complex128')
    for i in range(len(weights)):
        _sa_rdm += weights[i] * get_2_rdm(det_strings, cas, states[i], verbose)
    return _sa_rdm

def get_3_rdm_sa(det_strings, cas, states, weights, verbose):
    _sa_rdm = np.zeros((cas[1],cas[1],cas[1],cas[1],cas[1],cas[1]), dtype='complex128')
    for i in range(len(weights)):
        _sa_rdm += weights[i] * get_3_rdm(det_strings, cas, states[i], verbose)
    return _sa_rdm

def get_1_rdm(det_strings, cas, psi, verbose):
    _t0 = time.time()
    _rdm = np.zeros((cas[1],cas[1]), dtype='complex128')
    for i in range(len(det_strings)):
        occ_vec = bstring_to_occ_vec(det_strings[i], *cas)
        contrib = np.conjugate(psi[i])*psi[i]
        for p in occ_vec:
            _rdm[p, p] += contrib
        for j in range(len(det_strings)):
            if (get_excitation_level(det_strings[i], det_strings[j]) == 1):
                [[p], [q]], perm = get_excit_connection(det_strings[i], det_strings[j], 1, *cas)
                _rdm[p, q] += perm*np.conjugate(psi[i])*psi[j]
    _t1 = time.time()
    if (verbose): print(f'Time taken for 1-RDM build:  {(_t1-_t0):15.7f} s')
    return _rdm

def get_2_rdm(det_strings, cas, psi, verbose):
    _t0 = time.time()
    _rdm = np.zeros((cas[1],cas[1],cas[1],cas[1]), dtype='complex128')
    if (cas[0] < 2): return _rdm
    for i in range(len(det_strings)):
        # <p+ q+ q p>
        occ_vec = bstring_to_occ_vec(det_strings[i], *cas)
        # get all possible pairs of occupied spinorb
        contrib = np.conjugate(psi[i])*psi[i]
        for ip, p in enumerate(occ_vec):
            for q in occ_vec[:ip]:
                _rdm[p, q, p, q] += contrib
                _rdm[p, q, q, p] -= contrib
                _rdm[q, p, p, q] -= contrib
                _rdm[q, p, q, p] += contrib
        
        for j in range(len(det_strings)):
            exlvl = get_excitation_level(det_strings[i], det_strings[j])
            
            if (exlvl==1):
                # We need to accumulate all <p+ q+ q r>, it's sufficient to get the parity of <p+ r> since q+ q always cancel out
                [[p],[r]], perm = get_excit_connection(det_strings[i], det_strings[j], 1, *cas)
                f = annop(det_strings[i],p)[1]
                occ_vec = bstring_to_occ_vec(f, cas[0]-1, cas[1])
                contrib = perm*np.conjugate(psi[i])*psi[j]
                for q in occ_vec:
                    _rdm[p, q, r, q] += contrib
                    _rdm[p, q, q, r] -= contrib
                    _rdm[q, p, r, q] -= contrib
                    _rdm[q, p, q, r] += contrib
            elif (exlvl==2):
                # <p+ q+ s r>
                conn, perm = get_excit_connection(det_strings[i], det_strings[j], 2, *cas)
                p, q = conn[0] # get_excit_connection's perm is <q+ p+ r s>
                r, s = conn[1] # conn is in ascending order in spinor index, 
                contrib = perm*np.conjugate(psi[i])*psi[j]
                _rdm[p, q, r, s] += contrib
                _rdm[p, q, s, r] -= contrib
                _rdm[q, p, r, s] -= contrib
                _rdm[q, p, s, r] += contrib
    _t1 = time.time()
    if (verbose): print(f'Time taken for 2-RDM build:  {(_t1-_t0):15.7f} s')
    return _rdm

def get_3_rdm(det_strings, cas, psi, verbose):
    """
    gamma3^{pqr}_{stu} = <p+ q+ r+ u t s>
    """
    _t0 = time.time()
    _rdm = np.zeros((cas[1],cas[1],cas[1],cas[1],cas[1],cas[1]), dtype='complex128')
    if (cas[0] < 3): return _rdm
    for i in range(len(det_strings)):
        occ_vec = bstring_to_occ_vec(det_strings[i], *cas)
        # get all possible triplets of occupied spinors
        contrib = np.conjugate(psi[i])*psi[i]
        for ip, p in enumerate(occ_vec):
            for iq, q in enumerate(occ_vec[:ip]):
                for r in occ_vec[:iq]:
                    _rdm[p, q, r, p, q, r] += contrib
                    _rdm[q, r, p, p, q, r] += contrib
                    _rdm[r, p, q, p, q, r] += contrib
                    _rdm[p, r, q, p, q, r] -= contrib
                    _rdm[r, q, p, p, q, r] -= contrib
                    _rdm[q, p, r, p, q, r] -= contrib
                        
                    _rdm[p, q, r, q, r, p] += contrib
                    _rdm[q, r, p, q, r, p] += contrib
                    _rdm[r, p, q, q, r, p] += contrib
                    _rdm[p, r, q, q, r, p] -= contrib
                    _rdm[r, q, p, q, r, p] -= contrib
                    _rdm[q, p, r, q, r, p] -= contrib
                    
                    _rdm[p, q, r, r, p, q] += contrib
                    _rdm[q, r, p, r, p, q] += contrib
                    _rdm[r, p, q, r, p, q] += contrib
                    _rdm[p, r, q, r, p, q] -= contrib
                    _rdm[r, q, p, r, p, q] -= contrib
                    _rdm[q, p, r, r, p, q] -= contrib

                    _rdm[p, q, r, p, r, q] -= contrib
                    _rdm[q, r, p, p, r, q] -= contrib
                    _rdm[r, p, q, p, r, q] -= contrib
                    _rdm[p, r, q, p, r, q] += contrib
                    _rdm[r, q, p, p, r, q] += contrib
                    _rdm[q, p, r, p, r, q] += contrib
                    
                    _rdm[p, q, r, r, q, p] -= contrib
                    _rdm[q, r, p, r, q, p] -= contrib
                    _rdm[r, p, q, r, q, p] -= contrib
                    _rdm[p, r, q, r, q, p] += contrib
                    _rdm[r, q, p, r, q, p] += contrib
                    _rdm[q, p, r, r, q, p] += contrib
                    
                    _rdm[p, q, r, q, p, r] -= contrib
                    _rdm[q, r, p, q, p, r] -= contrib
                    _rdm[r, p, q, q, p, r] -= contrib
                    _rdm[p, r, q, q, p, r] += contrib
                    _rdm[r, q, p, q, p, r] += contrib
                    _rdm[q, p, r, q, p, r] += contrib
        
        for j in range(len(det_strings)):
            exlvl = get_excitation_level(det_strings[i], det_strings[j])
            
            if (exlvl==1):
                # We need to accumulate all <p+ q+ r+ r q s>, it's sufficient to get the parity of <p+ s> since q+r+ rq always cancel out
                [[p], [s]], perm = get_excit_connection(det_strings[i], det_strings[j], 1, *cas)
                f = annop(det_strings[i],p)[1]
                occ_vec = bstring_to_occ_vec(f, cas[0]-1, cas[1])
                contrib = perm*np.conjugate(psi[i])*psi[j]
                for iq, q in enumerate(occ_vec):
                    # q cannot be == r as violates exclusion principle
                    for r in occ_vec[:iq]:
                        _rdm[p, q, r, s, q, r] += contrib
                        _rdm[q, r, p, s, q, r] += contrib
                        _rdm[r, p, q, s, q, r] += contrib
                        _rdm[p, r, q, s, q, r] -= contrib
                        _rdm[r, q, p, s, q, r] -= contrib
                        _rdm[q, p, r, s, q, r] -= contrib

                        _rdm[p, q, r, q, r, s] += contrib
                        _rdm[q, r, p, q, r, s] += contrib
                        _rdm[r, p, q, q, r, s] += contrib
                        _rdm[p, r, q, q, r, s] -= contrib
                        _rdm[r, q, p, q, r, s] -= contrib
                        _rdm[q, p, r, q, r, s] -= contrib

                        _rdm[p, q, r, r, s, q] += contrib
                        _rdm[q, r, p, r, s, q] += contrib
                        _rdm[r, p, q, r, s, q] += contrib
                        _rdm[p, r, q, r, s, q] -= contrib
                        _rdm[r, q, p, r, s, q] -= contrib
                        _rdm[q, p, r, r, s, q] -= contrib

                        _rdm[p, q, r, s, r, q] -= contrib
                        _rdm[q, r, p, s, r, q] -= contrib
                        _rdm[r, p, q, s, r, q] -= contrib
                        _rdm[p, r, q, s, r, q] += contrib
                        _rdm[r, q, p, s, r, q] += contrib
                        _rdm[q, p, r, s, r, q] += contrib

                        _rdm[p, q, r, r, q, s] -= contrib
                        _rdm[q, r, p, r, q, s] -= contrib
                        _rdm[r, p, q, r, q, s] -= contrib
                        _rdm[p, r, q, r, q, s] += contrib
                        _rdm[r, q, p, r, q, s] += contrib
                        _rdm[q, p, r, r, q, s] += contrib

                        _rdm[p, q, r, q, s, r] -= contrib
                        _rdm[q, r, p, q, s, r] -= contrib
                        _rdm[r, p, q, q, s, r] -= contrib
                        _rdm[p, r, q, q, s, r] += contrib
                        _rdm[r, q, p, q, s, r] += contrib
                        _rdm[q, p, r, q, s, r] += contrib
            if (exlvl==2):
                # We need to accumulate all <p+ q+ r+ r t s>
                conn, perm = get_excit_connection(det_strings[i], det_strings[j], 2, *cas)
                p, q = conn[0] # get_excit_connection's perm is <q+ p+ r s>
                s, t = conn[1] # conn is in ascending order in spinor index, 
                f = annop_mult(det_strings[i],conn[0])[1]
                occ_vec = bstring_to_occ_vec(f, cas[0]-2, cas[1])
                contrib = perm*np.conjugate(psi[i])*psi[j]
                for r in occ_vec:                       
                    _rdm[p, q, r, s, t, r] += contrib
                    _rdm[q, r, p, s, t, r] += contrib
                    _rdm[r, p, q, s, t, r] += contrib
                    _rdm[p, r, q, s, t, r] -= contrib
                    _rdm[r, q, p, s, t, r] -= contrib
                    _rdm[q, p, r, s, t, r] -= contrib

                    _rdm[p, q, r, t, r, s] += contrib
                    _rdm[q, r, p, t, r, s] += contrib
                    _rdm[r, p, q, t, r, s] += contrib
                    _rdm[p, r, q, t, r, s] -= contrib
                    _rdm[r, q, p, t, r, s] -= contrib
                    _rdm[q, p, r, t, r, s] -= contrib

                    _rdm[p, q, r, r, s, t] += contrib
                    _rdm[q, r, p, r, s, t] += contrib
                    _rdm[r, p, q, r, s, t] += contrib
                    _rdm[p, r, q, r, s, t] -= contrib
                    _rdm[r, q, p, r, s, t] -= contrib
                    _rdm[q, p, r, r, s, t] -= contrib

                    _rdm[p, q, r, s, r, t] -= contrib
                    _rdm[q, r, p, s, r, t] -= contrib
                    _rdm[r, p, q, s, r, t] -= contrib
                    _rdm[p, r, q, s, r, t] += contrib
                    _rdm[r, q, p, s, r, t] += contrib
                    _rdm[q, p, r, s, r, t] += contrib

                    _rdm[p, q, r, r, t, s] -= contrib
                    _rdm[q, r, p, r, t, s] -= contrib
                    _rdm[r, p, q, r, t, s] -= contrib
                    _rdm[p, r, q, r, t, s] += contrib
                    _rdm[r, q, p, r, t, s] += contrib
                    _rdm[q, p, r, r, t, s] += contrib

                    _rdm[p, q, r, t, s, r] -= contrib
                    _rdm[q, r, p, t, s, r] -= contrib
                    _rdm[r, p, q, t, s, r] -= contrib
                    _rdm[p, r, q, t, s, r] += contrib
                    _rdm[r, q, p, t, s, r] += contrib
                    _rdm[q, p, r, t, s, r] += contrib
            if (exlvl==3):
                conn, perm = get_excit_connection(det_strings[i], det_strings[j], 3, *cas)
                p, q, r = conn[0]
                s, t, u = conn[1]
                
                contrib = perm*np.conjugate(psi[i])*psi[j]
                
                _rdm[p, q, r, s, t, u] += contrib
                _rdm[q, r, p, s, t, u] += contrib
                _rdm[r, p, q, s, t, u] += contrib
                _rdm[p, r, q, s, t, u] -= contrib
                _rdm[r, q, p, s, t, u] -= contrib
                _rdm[q, p, r, s, t, u] -= contrib

                _rdm[p, q, r, t, u, s] += contrib
                _rdm[q, r, p, t, u, s] += contrib
                _rdm[r, p, q, t, u, s] += contrib
                _rdm[p, r, q, t, u, s] -= contrib
                _rdm[r, q, p, t, u, s] -= contrib
                _rdm[q, p, r, t, u, s] -= contrib

                _rdm[p, q, r, u, s, t] += contrib
                _rdm[q, r, p, u, s, t] += contrib
                _rdm[r, p, q, u, s, t] += contrib
                _rdm[p, r, q, u, s, t] -= contrib
                _rdm[r, q, p, u, s, t] -= contrib
                _rdm[q, p, r, u, s, t] -= contrib

                _rdm[p, q, r, s, u, t] -= contrib
                _rdm[q, r, p, s, u, t] -= contrib
                _rdm[r, p, q, s, u, t] -= contrib
                _rdm[p, r, q, s, u, t] += contrib
                _rdm[r, q, p, s, u, t] += contrib
                _rdm[q, p, r, s, u, t] += contrib

                _rdm[p, q, r, u, t, s] -= contrib
                _rdm[q, r, p, u, t, s] -= contrib
                _rdm[r, p, q, u, t, s] -= contrib
                _rdm[p, r, q, u, t, s] += contrib
                _rdm[r, q, p, u, t, s] += contrib
                _rdm[q, p, r, u, t, s] += contrib

                _rdm[p, q, r, t, s, u] -= contrib
                _rdm[q, r, p, t, s, u] -= contrib
                _rdm[r, p, q, t, s, u] -= contrib
                _rdm[p, r, q, t, s, u] += contrib
                _rdm[r, q, p, t, s, u] += contrib
                _rdm[q, p, r, t, s, u] += contrib
    _t1 = time.time()
    if (verbose): print(f'Time taken for 3-RDM build:  {(_t1-_t0):15.7f} s')
    return _rdm

def enumerate_determinants(f0, nelec, norb, exlvl):
    occ_vec = bstring_to_occ_vec(f0, nelec, norb)
    unocc_vec = bstring_to_unocc_vec(f0, nelec, norb)

    nunocc = norb - nelec
    ndets = scipy.special.comb(nelec, exlvl, exact=True) * scipy.special.comb(nunocc, exlvl, exact=True)

    occ_excited = list(itertools.combinations(occ_vec,nelec-exlvl))
    unocc_excited = list(itertools.combinations(unocc_vec,exlvl))

    excited_occ_vecs = [sum(_, ()) for _ in list(itertools.product(occ_excited, unocc_excited))]

    det_strings = list(map(set_bit, excited_occ_vecs))

    assert len(det_strings) == ndets

    return det_strings

def get_H_IP(fi, pspace, psi, H1body, H2body, cas):
    """
    Evaluates <Phi_I|H|Psi_P> = V*
    """
    vj = 0.j
    for j, cj in enumerate(psi):
        vj += cj*get_hamil_element(fi, pspace[j], H1body, H2body, cas)

    return vj

def annihilate_state(psi, orb, basis):
    ann_psi = np.zeros_like(psi)
    ann_basis = np.zeros(len(basis), dtype='int')
    iann = 0
    for i in range(len(psi)):
        _ann_i = annop(basis[i], orb)
        if (_ann_i[0] != 0):
            ann_psi[iann] = _ann_i[0] * psi[i]
            ann_basis[iann] = _ann_i[1]
            iann += 1
    ann_psi = ann_psi[:iann]
    ann_basis = ann_basis[:iann]
    argsort = np.argsort(ann_basis)
    return ann_psi[argsort], ann_basis[argsort]

def eri_h5_write(mol, mo_coeffs, intor, erifile='tmp.h5', gaunt=False, terminal=False):
    phase = -1 if gaunt else 1
    pyscf.ao2mo.r_outcore.general(mol, mo_coeffs, erifile=erifile, dataname='tmp', intor=intor, aosym='s1')
    blksize = 400
    nij = mo_coeffs[0].shape[1] * mo_coeffs[1].shape[1]
    with h5py.File(erifile, mode='r+') as feri:
        for i0, i1 in pyscf.lib.prange(0, nij, blksize):
            buf = feri['mo_eri'][i0:i1]
            buf += phase*feri['tmp'][i0:i1]
            feri['mo_eri'][i0:i1] = buf
        
        if (terminal):
            del feri['tmp']

def clean_tmp():
    for i in glob.glob('tmp*'):
        os.remove(i)

def print_energies(energies, nstates=None, splitting=True):
    if (nstates is None): nstates = len(energies)

    print('{:<10}{:<20}{:<20}{:<20}{:<20}'.format('State','Energy / Eh','Splitting / Eh','Splitting / cm-1', 'Splitting / meV'))
    for istate in range(nstates):
        splitting = energies[istate] - energies[istate-1] if (istate>0) else 0
        print('{:<10}{:<20}{:<20}{:<20}{:<20}'.format(f'{istate:d}',f'{energies[istate]:+.7e}',f'{splitting:+.7e}',f'{splitting*EH_TO_WN:+.7e}',f'{splitting*EH_TO_EV*1000:+.7e}'))

class RelForte:
    def __init__(self, mol, c0=None, verbose=True, density_fitting=False, decontract=False):
        if (type(density_fitting) is bool):
            self.density_fitting = density_fitting
            self.df_basis = None
        elif(type(density_fitting) is str):
            self.density_fitting = True
            self.df_basis = density_fitting
        self.decontract = decontract
        if (self.decontract):
            self.mol, _ = mol.decontract_basis()
        else:
            self.mol = mol
        self.nuclear_repulsion = self.mol.energy_nuc()
        self.nelec = sum(self.mol.nelec)
        self.nocc = self.nelec
        self.nao = self.mol.nao
        self.nlrg = self.mol.nao*2
        self.nvirtual = self.nlrg - self.nocc
        self.verbose = verbose
        if (c0 is None):
            self.c0 = pyscf.lib.param.LIGHT_SPEED
        else:
            self.c0 = c0
            #pyscf.lib.param.LIGHT_SPEED = c0
        
    def run_rhf(self, transform=False, debug=True, frozen=None, dump_mo_coeff=None):
        _ti = time.time()
        if (self.verbose):
            print('='*47)
            print('{:^47}'.format('PySCF RHF interface'))
            print('='*47)

        if (self.density_fitting):
            if (self.verbose): print('{:#^47}'.format('Enabling density fitting!')) 
            self.rhf = pyscf.scf.RHF(self.mol).density_fit() if self.df_basis is None else pyscf.scf.RHF(self.mol).density_fit(self.df_basis)
        else:
            self.rhf = pyscf.scf.RHF(self.mol)

        self.rhf_energy = self.rhf.kernel()

        if (self.verbose): print(f"Non-relativistic RHF Energy: {self.rhf_energy:15.7f} Eh")

        if (dump_mo_coeff is not None):
            if dump_mo_coeff != False:
                if type(dump_mo_coeff) is str:
                    _fname = dump_mo_coeff
                else:
                    _fname = 'mo_coeff'
                if (self.verbose): print(f'Dumping MO coefficients to {_fname}')
                np.save(_fname, self.rhf.mo_coeff)

        _t1 = time.time()

        print(f'PySCF RHF time:              {_t1-_ti:15.7f} s')
        print('-'*47)
        
        if (transform):
            self.nonrel_ao2mo(self.rhf.mo_coeff, frozen)

            if (debug and self.verbose):
                self.rhf_e1 = np.einsum('ii->',self.rhf_hcore_spinorb[:self.nocc, :self.nocc])
                self.rhf_e2 = 0.5*np.einsum('ijij->',self.rhf_eri_full_asym[:self.nocc, :self.nocc, :self.nocc, :self.nocc])            
                self.rhf_e_rebuilt = self.rhf_e1 + self.rhf_e2 + self.nuclear_repulsion
                print(f"Rebuilt RHF Energy:          {self.rhf_e_rebuilt.real:15.7f} Eh")
                print(f"Error to PySCF:              {np.abs(self.rhf_e_rebuilt.real - self.rhf_energy):15.7f} Eh")
                if (frozen is None): print(f"Diff 1e:                     {np.abs(self.rhf.scf_summary['e1']-self.rhf_e1):15.7f} Eh")
                if (frozen is None): print(f"Diff 2e:                     {np.abs(self.rhf.scf_summary['e2']-self.rhf_e2):15.7f} Eh")
                print('-'*47)
            
            if (self.verbose):
                _tf = time.time()
                print(f'RHF time:                    {_tf-_ti:15.7f} s')
                print('='*47)
        
    def run_dhf(self, transform=False, debug=False, frozen=None, with_gaunt=False, with_breit=False, with_ssss=True, dump_mo_coeff=None, algo='disk', erifile=None, fake_dhf=None):
        _ti = time.time()

        if (self.verbose):
            print('='*47)
            print('{:^47}'.format('PySCF DHF interface'))
            print('='*47)

        # Run relativistic Dirac-Hartree-Fock
        if (self.density_fitting):
            if (self.verbose): print('{:#^47}'.format('Enabling density fitting!')) 
            self.dhf = pyscf.scf.DHF(self.mol).density_fit() if self.df_basis is None else pyscf.scf.DHF(self.mol).density_fit(self.df_basis)
        else:
            self.dhf = pyscf.scf.DHF(self.mol)

        self.dhf.with_gaunt = with_gaunt
        self.dhf.with_breit = with_breit
        self.dhf.with_ssss = with_ssss
        
        if (type(fake_dhf) is str):
            f = h5py.File(fake_dhf, 'r')
            self.dhf_energy = f['SCF']['TOTAL_ENERGY'][:][0]
        else:
            self.dhf_energy = self.dhf.kernel()
            if (self.verbose): 
                _t0 = time.time()
                print(f"Relativistic DHF Energy:     {self.dhf_energy:15.7f} Eh")
                print(f'PySCF RHF time:              {_t0-_ti:15.7f} s')
                print('-'*47)

            if (dump_mo_coeff is not None):
                if dump_mo_coeff != False:
                    if type(dump_mo_coeff) is str:
                        _fname = dump_mo_coeff
                    else:
                        _fname = 'mo_coeff'
                    if (self.verbose): print(f'Dumping MO coefficients to {_fname}')
                    np.save(_fname, self.dhf.mo_coeff)

            if (transform):
                self.rel_ao2mo(self.dhf.mo_coeff, frozen, algo, erifile)
                
                self.dhf_e1 = np.einsum('ii->',self.dhf_hcore_mo[:self.nocc, :self.nocc])
                self.dhf_e2 = 0.5*np.einsum('ijij->',self.dhf_eri_full_asym[:self.nocc, :self.nocc, :self.nocc, :self.nocc])            
                self.dhf_e_rebuilt = self.dhf_e1 + self.dhf_e2 + self.nuclear_repulsion
                if (debug and self.verbose):
                    print(f"Rebuilt DHF Energy:          {self.dhf_e_rebuilt.real:15.7f} Eh")
                    print(f"Error to PySCF:              {np.abs(self.dhf_e_rebuilt.real - self.dhf_energy):15.7f} Eh")
                    if (frozen is None):
                        print(f"Diff 1e:                     {np.abs(self.dhf.scf_summary['e1']-self.dhf_e1):15.7f} Eh")
                        print(f"Diff 2e:                     {np.abs(self.dhf.scf_summary['e2']-self.dhf_e2):15.7f} Eh")
                
                _t3 = time.time()
                
                if (self.verbose):
                    print(f'Total time taken:            {(_t3-_t0):15.7f} s')
                    print('='*47)
    
    def do_mmfx2c(self, mo_coeff, mode='orthonormal'):
        if (mode == 'orthonormal'):
            _s, _U_ovlp = np.linalg.eigh(self.dhf.get_ovlp()) # Diagonalize overlap matrix
            _X_lowdin = _U_ovlp @ np.diag(_s**(-0.5)) # The Lowdin canonicalization matrix
            _Xinv = np.diag(np.sqrt(_s)) @ _U_ovlp.T.conj()

            assert (np.allclose(_X_lowdin @ _Xinv, np.eye(self.norb)))

            _F_4c = _X_lowdin.T.conj() @ self.dhf.get_fock() @ _X_lowdin
            _e, _mo_coeff_on = np.linalg.eigh(_F_4c)
            assert np.allclose(self.dhf.mo_energy, _e)

            _cp_neg = _mo_coeff_on[self.nlrg:, :self.nlrg]
            _cl_neg = _mo_coeff_on[:self.nlrg, :self.nlrg]
            _Xtrans = np.linalg.solve(_cp_neg@_cp_neg.T.conj(), -_cp_neg@_cl_neg.T.conj())

            _W1 = np.block([[np.eye(self.nlrg), -_Xtrans.T.conj()],[_Xtrans, np.eye(self.nlrg)]])
            _W2 = np.block([[np.diag((1+np.diag(_Xtrans.T.conj()@_Xtrans))**(-0.5)) , np.zeros((self.nlrg, self.nlrg))],\
                        [np.zeros((self.nlrg, self.nlrg)), np.diag(((1+np.diag(_Xtrans@_Xtrans.T.conj()))**(-0.5)))]])
            self.U_x2c = _W1 @ _W2
            print(np.abs(self.U_x2c).sum())

            _F_2c = (self.U_x2c.T.conj() @ _F_4c @ self.U_x2c)[:self.nlrg, :self.nlrg]
            _e_2c, _C_2c_on = np.linalg.eigh(_F_2c)

            _X_lowdin_2c = (_X_lowdin @ self.U_x2c)[:self.nlrg, :self.nlrg]
            return _X_lowdin_2c @ _C_2c_on
        elif (mode == 'atomic'):
            _cp_neg = mo_coeff[self.nlrg:, :self.nlrg]
            _cl_neg = mo_coeff[:self.nlrg, :self.nlrg]
            _Xtrans = np.linalg.solve(_cp_neg@_cp_neg.T.conj(), -_cp_neg@_cl_neg.T.conj())

            _W1 = np.block([[np.eye(self.nlrg), -_Xtrans.T.conj()],[_Xtrans, np.eye(self.nlrg)]])
            _W2 = np.block([[np.diag((1+np.diag(_Xtrans.T.conj()@_Xtrans))**(-0.5)) , np.zeros((self.nlrg, self.nlrg))],\
                        [np.zeros((self.nlrg, self.nlrg)), np.diag(((1+np.diag(_Xtrans@_Xtrans.T.conj()))**(-0.5)))]])
            self.U_x2c = _W1 @ _W2
            return (self.U_x2c.T.conj() @ mo_coeff)[:self.nlrg, self.nlrg:]
    

    def run_mp2(self, relativistic=True):
        if (relativistic):
            _hcore = self.dhf_hcore_mo
            _eri = self.dhf_eri_full_asym
            _method = 'Relativistic'
            _e_scf = self.dhf_energy
        else:
            _hcore = self.rhf_hcore_spinorb
            _eri = self.rhf_eri_full_asym
            _method = 'Non-relativistic'
            _e_scf = self.rhf_energy
        
        if (self.verbose):
            print('')
            print('='*47)
            print('{:^47}'.format(f'{_method} MP2'))
            print('='*47)

        _t0 = time.time()
        _upq = np.einsum('piqi->pq',_eri[:,:self.nocc,:,:self.nocc])
        self.fock_mo = _upq + _hcore

        self.D2 = np.zeros((self.nocc,self.nocc,self.nvirtual,self.nvirtual))
        _e = np.diag(self.fock_mo)
        for i in range(self.nocc):
            for j in range(self.nocc):
                for a in range(self.nvirtual):
                    for b in range(self.nvirtual):
                        self.D2[i,j,a,b] = -1./((_e[a+self.nocc] + _e[b+self.nocc] - _e[i] - _e[j]).real)

        _oovv = _eri[:self.nocc,:self.nocc,self.nocc:,self.nocc:]
        _vvoo = _eri[self.nocc:,self.nocc:,:self.nocc,:self.nocc]

        self.e_mp2 = 0.25*np.einsum('ijab,ijab,abij->',_oovv,self.D2,_vvoo)

        try:
            assert(abs(self.e_mp2.imag) < MACHEPS)
        except AssertionError:
            print(f'Imaginary part of MP2 energy is larger than {MACHEPS}')
        
        self.e_mp2 = self.e_mp2.real

        if (not relativistic):
            self.nonrel_mp2 = pyscf.mp.MP2(self.rhf)
            _e_mp2 = self.nonrel_mp2.kernel()[0]

        _t1 = time.time()
        if (self.verbose):
            print(f'MP2 Ecorr:                   {self.e_mp2.real:15.7f} Eh')
            print(f'MP2 Energy:                  {(self.e_mp2 + _e_scf).real:15.7f} Eh')
            if (not relativistic): print(f'Error to PySCF:              {(self.e_mp2-_e_mp2):15.7f} Eh')
            print(f'Time taken:                  {(_t1-_t0):15.7f} s')
            print('='*47)
    
    def run_casscf_pyscf(self, cas, transform=False, frozen=None):
        _ti = time.time()
        if (self.verbose):
            print('='*47)
            print('{:^47}'.format('PySCF CASSCF interface'))
            print('='*47)
        
        casscf = pyscf.mcscf.CASSCF(self.rhf, int(cas[1]//2), cas[0])
        res = casscf.kernel()
        self.e_casscf = res[0]

        if (transform):
            self.nonrel_ao2mo(res[3], frozen=None)

            if (self.verbose):
                _tf = time.time()
                print(f'CASSCF time:                 {_tf-_ti:15.7f} s')
                print('='*47)
    
    def run_casci(self, cas=None, do_fci=False, rdm_level=0, relativistic=True, semi_canonicalize=True, state_avg=None, sa_weights=None):
        _t0 = time.time()
        
        try:
            assert ((cas is None) and do_fci) or ((cas is not None) and (not do_fci))
        except AssertionError:
            raise Exception("If not doing FCI then a CAS must be provided via the 'cas' argument!")

        if (state_avg is not None):
            try:
                assert (type(state_avg) is list)
            except AssertionError:
                raise Exception("state_avg must be a list of CASCI states to average over!")
            
            try:
                assert (type(sa_weights) is list and len(sa_weights) == len(state_avg))
            except AssertionError:
                raise Exception("sa_weights must be a list of the same length as state_avg!")

            try:
                assert (np.isclose(sum(sa_weights), 1.0))
            except AssertionError:
                raise Exception("sa_weights doesn't add up to 1.0!")
        else:
            state_avg = [0]
            sa_weights = [1.0]

        self.state_avg = state_avg
        self.sa_weights = sa_weights
        
        if (do_fci):
            cas = (self.nelec, self.nlrg)

        self.cas = cas

        if (type(cas) is tuple):
            if (type(cas[0]) is int):
                # Using a CAS
                try:
                    assert int(cas[0]) <= int(cas[1])
                except AssertionError:
                    raise Exception("Number of CAS electrons must be <= number of CAS spinors!")
                
                self.ncore = self.nelec - self.cas[0]
                self.nact = self.cas[1]
                self.nvirt = self.nlrg-(self.ncore+self.nact) # This is different from nvirtual, which is in the single-reference sense (nvirt in the HF reference)
                self.nhole = self.ncore+self.nact
                self.npart = self.nact+self.nvirt

                self.core = slice(0,self.ncore)
                self.active = slice(self.ncore, self.ncore+self.nact)
                self.virt = slice(self.ncore+self.nact, self.nlrg)
                self.hole = slice(0,self.ncore+self.nact)
                self.part = slice(self.ncore, self.nlrg)
            elif (type(cas[0]) is list):
                # Using a GAS: cas = (occ_orbs, active_orbs)
                try:
                    assert len(cas[1]) + len(cas[2]) > self.nelec
                except AssertionError:
                    raise Exception("Number of frozen + active spinorbitals <= number of electrons!")

                self.ncore = len(self.cas[0])
                self.nact = len(self.cas[1])
                self.nvirt = self.nlrg-(self.ncore+self.nact) # This is different from nvirtual, which is in the single-reference sense (nvirt in the HF reference)
                self.nhole = self.ncore+self.nact
                self.npart = self.nact+self.nvirt

                self.core = self.cas[0]
                self.active = self.cas[1]
                self.virt = list(np.sort(list(set(list(range(self.nlrg))) - set(self.core) - set(self.active))))
                self.hole = list(np.sort(self.core + self.active))
                self.part = list(np.sort(self.active + self.virt))
                
            self.hc = self.core
            self.ha = self.active
            self.pa = slice(0,self.nact)
            self.pv = slice(self.nact,self.nact+self.nvirt)

            self.hh = self.hole
            self.pp = slice(0,self.npart)
        else:
            raise Exception("'cas' must be a tuple of two integers or orbital lists!")
                
        self.semi_can = semi_canonicalize
        
        self.e_casci_frzc = 0.0
            
        if (relativistic):
            _hcore = self.dhf_hcore_mo
            _eri = self.dhf_eri_full_asym
            _method = 'Relativistic'
            _e_scf = self.dhf_energy
        else:
            _hcore = self.rhf_hcore_spinorb
            _eri = self.rhf_eri_full_asym
            _method = 'Non-relativistic'
            _e_scf = self.rhf_energy

        if (self.verbose):
            print('')
            print('='*47)
            print('{:^47}'.format(f'{_method} CASCI({self.cas[0]},{self.cas[1]})'))
            print('='*47)
            
        if (self.ncore != 0):
            self.e_casci_frzc = np.einsum('ii->',_hcore[self.core,self.core]) + 0.5*np.einsum('ijij->',_eri[self.core,self.core,self.core,self.core])
            _hcore_frzc_cas = _hcore[self.active, self.active].copy() + np.einsum('ipjp->ij',_eri[self.active,self.core,self.active,self.core])
            _hcore_cas = _hcore_frzc_cas
        else:
            _hcore_cas = _hcore
            
        self.ncombs, self.det_strings = form_cas_determinant_strings(*self.cas)
        self.cas_hamil = form_cas_hamiltonian(_hcore_cas, _eri, self.det_strings, self.verbose, self.cas, ncore=self.ncore)

        _t1 = time.time()
        
        self.casci_eigvals, self.casci_eigvecs = np.linalg.eigh(self.cas_hamil)
        _t2 = time.time()
                
        if (rdm_level > 0):
            try:
                assert rdm_level <= 3
            except AssertionError:
                raise Exception("RDM level up to 3 supported!")
                
            _rdms = {'max_rdm_level':rdm_level}
            if (rdm_level>=1):
                _psi = [self.casci_eigvecs[:,i] for i in self.state_avg]
                _rdms['1rdm'] = get_1_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)
                # J. Chem. Phys. 146, 124132 (2017), eq. 18
                self.gen_fock_canon = _hcore.copy() + np.einsum('piqi->pq',_eri[:,self.core,:,self.core]) + np.einsum('piqj,ij->pq',_eri[:,self.active,:,self.active],_rdms['1rdm'])
                self.fock = self.gen_fock_canon
                if (semi_canonicalize):
                    _gen_fock_diag = np.zeros_like(self.gen_fock_canon)
                    _gen_fock_diag[self.core,self.core] = self.gen_fock_canon[self.core,self.core]
                    _gen_fock_diag[self.active,self.active] = self.gen_fock_canon[self.active,self.active]
                    _gen_fock_diag[self.virt,self.virt] = self.gen_fock_canon[self.virt,self.virt]
                    self.gen_fock_eigvals, self.semicanonicalizer = np.linalg.eigh(_gen_fock_diag)
                    self.gen_fock_semicanon = np.einsum('ip,ij,jq->pq',np.conj(self.semicanonicalizer), self.gen_fock_canon, self.semicanonicalizer)
                    self.fock = self.gen_fock_semicanon
                    self.F0 = np.diag(np.diag(self.fock))
                    self.F1 = np.copy(self.fock - self.F0)
                    self.semicanonicalizer_active = self.semicanonicalizer[self.active, self.active]
                else:
                    self.semicanonicalizer = np.diag((np.zeros(self.fock.shape[0],dtype='complex128')+1.0))
                    self.semicanonicalizer_active = self.semicanonicalizer[self.active, self.active]
            if (rdm_level>=2):
                if (self.cas[0]>=2):
                    _rdms['2rdm'] = get_2_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)
                else:
                    _rdms['2rdm'] = np.zeros((self.cas[1],self.cas[1],self.cas[1],self.cas[1]), dtype='complex128')
            if (rdm_level>=3):
                if (self.cas[0]>=3):
                    _rdms['3rdm'] = get_3_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)
                else:
                    _rdms['3rdm'] = np.zeros((self.cas[1],self.cas[1],self.cas[1],self.cas[1],self.cas[1],self.cas[1]), dtype='complex128')

            if (semi_canonicalize):
                # a_p+~ = a_q+ U _qp
                # a_p ~ = a_q  U*_qp
                # <a_p+~ a_q~> = <a_i+ U_ip a_j U*_jq> = U_ip gamma_ij U*_jq
                # cf. Helgaker Ch. 3.2
                _rdms_semican = {'max_rdm_level':rdm_level}
                if (rdm_level>=1): _rdms_semican['1rdm'] = np.einsum('ip,ij,jq->pq', (self.semicanonicalizer_active), _rdms['1rdm'], np.conj(self.semicanonicalizer_active), optimize='optimal')
                if (rdm_level>=2): _rdms_semican['2rdm'] = np.einsum('ip,jq,ijkl,kr,ls->pqrs', (self.semicanonicalizer_active), self.semicanonicalizer_active, _rdms['2rdm'], np.conj(self.semicanonicalizer_active),np.conj(self.semicanonicalizer_active), optimize='optimal')
                if (rdm_level>=3): _rdms_semican['3rdm'] = np.einsum('ip,jq,kr,ijklmn,ls,mt,nu->pqrstu', (self.semicanonicalizer_active), self.semicanonicalizer_active, self.semicanonicalizer_active, _rdms['3rdm'], np.conj(self.semicanonicalizer_active),np.conj(self.semicanonicalizer_active), np.conj(self.semicanonicalizer_active), optimize='optimal')
                _eri = np.einsum('ip,jq,ijkl,kr,ls->pqrs',np.conj(self.semicanonicalizer),np.conj(self.semicanonicalizer),_eri,self.semicanonicalizer,self.semicanonicalizer,optimize='optimal')
        _t3 = time.time()
        
        _sa_casci_eigvals = np.dot(self.casci_eigvals[self.state_avg], self.sa_weights)

        if (not do_fci):
            if (self.verbose): print(f'E_frzc:                      {self.e_casci_frzc.real:15.7f} Eh')
            if (self.verbose): print(f'E_cas:                       {self.casci_eigvals[0]:15.7f} Eh')
            if (self.verbose): print(f'E_nuc:                       {self.nuclear_repulsion:15.7f} Eh')
            self.e_casci = self.e_casci_frzc.real+_sa_casci_eigvals.real+self.nuclear_repulsion
            if (self.verbose): print(f'E_casci:                     {self.e_casci:15.7f} Eh')
        else:
            self.e_casci = _sa_casci_eigvals.real+self.nuclear_repulsion
            if (self.verbose): print(f'E_casci:                     {self.e_casci:15.7f} Eh')

        try:
            assert(abs(self.e_casci.imag) < MACHEPS)
        except AssertionError:
            print(f'Imaginary part of CASCI energy is larger than {MACHEPS}')
        
        self.e_casci = self.e_casci_save = self.e_casci.real

        if (rdm_level >= 2):
            _Eref_test = self.nuclear_repulsion
            if (semi_canonicalize):
                _hcore = np.einsum('ip,ij,jq->pq',np.conj(self.semicanonicalizer),_hcore,(self.semicanonicalizer))
            _Eref_test += np.einsum('mm->',_hcore[self.core,self.core])
            _Eref_test += 0.5 * np.einsum('mnmn->',_eri[self.core,self.core,self.core,self.core])

            _Eref_test += np.einsum('mumv,uv->',_eri[self.core,self.active,self.core,self.active],_rdms_semican['1rdm'])

            _Eref_test += np.einsum('uv,uv->',_hcore[self.active,self.active],_rdms_semican['1rdm'])
            _Eref_test += 0.25 * np.einsum('uvxy,uvxy->',_eri[self.active,self.active,self.active,self.active],_rdms_semican['2rdm'])
            self.e_casci_rebuilt = _Eref_test.real
            if (self.verbose): print(f'E0 (from RDM):               {self.e_casci_rebuilt:15.7f} Eh')
        if (self.verbose): print(f'Ecorr:                       {self.e_casci-_e_scf.real:15.7f} Eh')
                
        _t4 = time.time()
        
        if (not relativistic):
            self.nonrel_casci = pyscf.mcscf.CASCI(self.rhf, int(self.cas[1]/2), self.cas[0])
            self.e_casci_pyscf = self.nonrel_casci.kernel()[0]
            if (self.verbose): print(f'Error to PySCF:              {(self.e_casci-self.e_casci_pyscf):15.7f} Eh')
        
        if (self.verbose):
            print()
            print('Timing summary')
            print(f'... Hamil build:              {(_t1-_t0):15.7f} s')
            print(f'... Hamil diag:               {(_t2-_t1):15.7f} s')

        if (relativistic):
            self.dhf_hcore_mo = _hcore
            self.dhf_eri_full_asym = _eri
        else:
            self.rhf_hcore_spinorb = _hcore
            self.rhf_eri_full_asym = _eri
            
        if (rdm_level > 0):
            if (self.verbose): print(f'... RDM build:                {(_t3-_t2):15.7f} s')
            self.rdms_canon = _rdms
            self.rdms = _rdms_semican
        if (self.verbose):
            if (len(self.state_avg) > 1):
                print_energies(np.real(self.casci_eigvals[self.state_avg]+self.e_casci_frzc+self.nuclear_repulsion))
            print(f'Total time taken:             {(_t4-_t0):15.7f} s')
            print('='*47)

    def form_denominators(self, s):
        fdiag = np.real(np.diagonal(self.fock))
        self.d1 = np.zeros((self.nhole,self.npart),dtype='float64')
        self.d2 = np.zeros((self.nhole,self.nhole,self.npart,self.npart),dtype='float64')
        self.delta1 = np.zeros((self.nhole,self.npart),dtype='float64')
        self.delta2 = np.zeros((self.nhole,self.nhole,self.npart,self.npart),dtype='float64')
        for i in range(self.nhole):
            for k in range(self.npart):
                self.d1[i,k] = regularized_denominator(fdiag[i]-fdiag[k+self.ncore], s)
                self.delta1[i,k] = fdiag[i]-fdiag[k+self.ncore]
                for j in range(self.nhole):
                    for l in range(self.npart):
                        self.d2[i,j,k,l] = regularized_denominator(fdiag[i]+fdiag[j]-fdiag[k+self.ncore]-fdiag[l+self.ncore], s)
                        self.delta2[i,j,k,l] = fdiag[i]+fdiag[j]-fdiag[k+self.ncore]-fdiag[l+self.ncore]
                            
        self.denom_act = np.zeros((self.nact,self.nact),dtype='float64')
        for i in range(self.nact):
            for j in range(self.nact):
                self.denom_act[i,j] = (fdiag[i+self.ncore]-fdiag[j+self.ncore])
                
        self.d1_exp = np.zeros((self.nhole,self.npart),dtype='float64')
        self.d2_exp = np.zeros((self.nhole,self.nhole,self.npart,self.npart),dtype='float64')
        for i in range(self.nhole):
            for k in range(self.npart):
                self.d1_exp[i,k] = np.exp(-s*(fdiag[i]-fdiag[k+self.ncore])**2)
                for j in range(self.nhole):
                    for l in range(self.npart):
                        self.d2_exp[i,j,k,l] = np.exp(-s*(fdiag[i]+fdiag[j]-fdiag[k+self.ncore]-fdiag[l+self.ncore])**2)

    def form_amplitudes(self, F, V):
        T2 = V[self.hole,self.hole,self.part,self.part].copy() * self.d2
        T2[self.ha,self.ha,self.pa,self.pa] = .0j

        T1 = F[self.hole,self.part].copy()
        T1 += np.einsum('xu,iuax,xu->ia', self.denom_act, T2[:,self.ha,:,self.pa],self.cumulants['gamma1'])
        T1 *= self.d1
        T1[self.ha,self.pa] = .0j

        return T1, T2
    
    def run_dsrg_mrpt3(self, s, relativistic, relax=None, relax_convergence=1e-8, maxiter=20, pt2_only=False):
        self.relax = relax
        if (relativistic):           
            _eri = self.dhf_eri_full_asym
            _method = 'Relativistic'
            _hcore = self.dhf_hcore_mo
        else:
            _eri = self.rhf_eri_full_asym
            _method = 'Non-relativistic'
            _hcore = self.rhf_hcore_spinorb

        if (self.verbose):
            print('')
            print('='*47)
            print('{:^47}'.format(f'{_method} DSRG-MRPT3'))
            print('='*47)

        _t0 = time.time()

        if (relax is None):
            nrelax = 0
        elif (relax == 'once'):
            nrelax = 1
        elif (relax == 'twice'):
            nrelax = 2
        elif (relax == 'iterate'):
            nrelax = maxiter
        elif (type(relax) is int):
            if (relax < 0):
                raise Exception(f'Relax iteration must be positive!')
            else:
                nrelax = min(maxiter, relax)
        else:
            raise Exception(f'Relax option {relax} is not implemented yet!')
        
        self.relax_ref = nrelax > 0

        _t2 = time.time()
        self.converged = False
        self.dsrg_mrpt3_update(s, _eri, pt2_only) # We need to do mrpt3 at least once
        _verbose = self.verbose

        if (nrelax > 0):
            self.relax_energies = np.zeros((maxiter,3)) # [iter, [unrelaxed, relaxed, Eref]]
            
            self.verbose = False
            if (_verbose):
                print('-'*47)
                print('{:^47}'.format(f'DSRG-MRPT3 reference relaxation'))
                print('-'*47)
                print('{:<30}{:<20}{:<10}'.format('Iter','Energy','Delta E'))
                print(f'   -Eref  {self.e_casci:.7f}       ')
                print(f'   -Ecorr {self.e_dsrg_mrpt3.real:.7f}       ')
        else:
            self.relax_energies = np.zeros((1,3))
            self.relax_energies[0, 0] = self.e_dsrg_mrpt3.real+self.e_casci
            self.relax_energies[0, 2] = self.e_casci
        
        for irelax in range(nrelax):
            self.relax_energies[irelax, 0] = self.e_dsrg_mrpt3.real+self.e_casci
            self.relax_energies[irelax, 2] = self.e_casci

            if (_verbose): print('{:<30}{:<20}{:<10}'.format(f'<Psi^({irelax:d})|Hbar^({irelax:d})|Psi^({irelax:d})>',f'{self.e_dsrg_mrpt3.real+self.e_casci:.7f}',f'{self.relax_energies[irelax][0]-self.relax_energies[irelax-1][1]:.5e}'))

            if (nrelax == 2 and irelax == 1): break
            self.dsrg_mrpt3_reference_relaxation(_eri)
            self.relax_energies[irelax, 1] = self.e_dsrg_mrpt3_relaxed
            if (_verbose): print(f'   -Erelax{self.e_relax:.7f}       ')
            if (_verbose): print('{:<30}{:<20}{:<10}'.format(f'<Psi^({irelax:d})|Hbar^({irelax+1:d})|Psi^({irelax:d})>',f'{self.e_dsrg_mrpt3_relaxed:.7f}',f'{self.relax_energies[irelax][1]-self.relax_energies[irelax][0]:.5e}'))
            if (self.test_relaxation_convergence(irelax, relax_convergence)): break
            if (nrelax == 1): break

            _psi = [self.dsrg_mrpt3_relax_eigvecs[:,i] for i in self.state_avg]
            self.rdms_canon['1rdm'] = get_1_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)
            self.rdms_canon['2rdm'] = get_2_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)
            self.rdms_canon['3rdm'] = get_3_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)

            _eri_canon = np.einsum('ip,jq,pqrs,kr,ls->ijkl',np.conjugate(self.semicanonicalizer),np.conjugate(self.semicanonicalizer),_eri,self.semicanonicalizer,self.semicanonicalizer,optimize='optimal')
            _hcore_canon = np.einsum('ip,pq,jq->ij',np.conjugate(self.semicanonicalizer),_hcore,self.semicanonicalizer,optimize='optimal')
            del _eri, _hcore

            _gen_fock_canon = _hcore_canon.copy() + np.einsum('piqi->pq',_eri_canon[:,self.core,:,self.core]) + np.einsum('piqj,ij->pq',_eri_canon[:,self.active,:,self.active],self.rdms_canon['1rdm'])

            _gen_fock_diag = np.zeros(_gen_fock_canon.shape, dtype='complex128')
            _gen_fock_diag[self.core,self.core] = _gen_fock_canon[self.core,self.core].copy()
            _gen_fock_diag[self.active,self.active] = _gen_fock_canon[self.active,self.active].copy()
            _gen_fock_diag[self.virt,self.virt] = _gen_fock_canon[self.virt,self.virt].copy()
            _, self.semicanonicalizer = np.linalg.eigh(_gen_fock_diag)
            self.gen_fock_semicanon = np.einsum('ip,ij,jq->pq',np.conj(self.semicanonicalizer), _gen_fock_canon, self.semicanonicalizer)
            self.fock = self.gen_fock_semicanon
            self.F0 = np.diag(np.diag(self.fock))
            self.F1 = np.copy(self.fock - self.F0)
            self.semicanonicalizer_active = self.semicanonicalizer[self.active, self.active]

            self.rdms['1rdm'] = np.einsum('ip,ij,jq->pq', self.semicanonicalizer_active, self.rdms_canon['1rdm'], np.conj(self.semicanonicalizer_active), optimize='optimal')
            self.rdms['2rdm'] = np.einsum('ip,jq,ijkl,kr,ls->pqrs', self.semicanonicalizer_active, self.semicanonicalizer_active, self.rdms_canon['2rdm'], np.conj(self.semicanonicalizer_active),np.conj(self.semicanonicalizer_active), optimize='optimal')
            self.rdms['3rdm'] = np.einsum('ip,jq,kr,ijklmn,ls,mt,nu->pqrstu', self.semicanonicalizer_active, self.semicanonicalizer_active, self.semicanonicalizer_active, self.rdms_canon['3rdm'], np.conj(self.semicanonicalizer_active),np.conj(self.semicanonicalizer_active), np.conj(self.semicanonicalizer_active), optimize='optimal')
            _eri_semican = np.einsum('ip,jq,ijkl,kr,ls->pqrs',np.conj(self.semicanonicalizer),np.conj(self.semicanonicalizer),_eri_canon,self.semicanonicalizer,self.semicanonicalizer,optimize='optimal')
            _hcore_semican = np.einsum('ip,ij,jq->pq',np.conj(self.semicanonicalizer),_hcore_canon,self.semicanonicalizer,optimize='optimal')
            _eri = _eri_semican
            _hcore = _hcore_semican
            del _eri_semican, _hcore_semican, _eri_canon, _hcore_canon

            self.e_casci = self.nuclear_repulsion
            self.e_casci += np.einsum('mm->',_hcore[self.core,self.core])
            self.e_casci += 0.5 * np.einsum('mnmn->',_eri[self.core,self.core,self.core,self.core])
            self.e_casci += np.einsum('mumv,uv->',_eri[self.core,self.active,self.core,self.active],self.rdms['1rdm'])
            self.e_casci += np.einsum('uv,uv->',_hcore[self.active,self.active],self.rdms['1rdm'])
            self.e_casci += 0.25 * np.einsum('uvxy,uvxy->',_eri[self.active,self.active,self.active,self.active],self.rdms['2rdm'])

            self.e_casci = self.e_casci.real
            if (_verbose): print(f'   -Eref  {self.e_casci:.7f}       ')
            
            self.dsrg_mrpt3_update(s, _eri, pt2_only)
            if (_verbose): print(f'   -Ecorr {self.e_dsrg_mrpt3.real:.7f}       ')
            
        self.verbose = _verbose

        _t3 = time.time()

        try:
            assert(abs(self.e_dsrg_mrpt3.imag) < MACHEPS)
        except AssertionError:
            print(f'Imaginary part of DSRG-MRPT3 energy, {self.e_dsrg_mrpt3.imag} is larger than {MACHEPS}')
        
        self.e_dsrg_mrpt3 = self.e_dsrg_mrpt3.real

        _t1 = time.time()

        if (self.verbose):
            print(f'Unrelaxed DSRG-MRPT3 energy:           {self.relax_energies[0][0]:15.7f} Eh')
            print(f'Unrelaxed DSRG-MRPT3 E_corr:           {self.relax_energies[0][0]-self.relax_energies[0][2]:15.7f} Eh')
            self.e_dsrg_mrpt3_unrelaxed = self.relax_energies[0][0]
            self.e_dsrg_mrpt3 = self.e_dsrg_mrpt3_unrelaxed
            if (nrelax > 0):
                print(f'Partially relaxed DSRG-MRPT3 energy:   {self.relax_energies[0][1]:15.7f} Eh')
                print(f'Partially relaxed DSRG-MRPT3 E_corr:   {self.relax_energies[0][1]-self.relax_energies[0][2]:15.7f} Eh')
                self.e_dsrg_mrpt3_partially_relaxed = self.relax_energies[0][1]
                self.e_dsrg_mrpt3 = self.e_dsrg_mrpt3_partially_relaxed
            if (nrelax > 1):
                print(f'Relaxed DSRG-MRPT3 energy:             {self.relax_energies[1][0]:15.7f} Eh')
                print(f'Relaxed DSRG-MRPT3 E_corr:             {self.relax_energies[1][0]-self.relax_energies[1][2]:15.7f} Eh')
                self.e_dsrg_mrpt3_relaxed = self.relax_energies[1][0]
                self.e_dsrg_mrpt3 = self.e_dsrg_mrpt3_relaxed
            if (nrelax > 2):
                print(f'Fully relaxed DSRG-MRPT3 energy:       {self.relax_energies[irelax][0]:15.7f} Eh')
                print(f'Fully relaxed DSRG-MRPT3 E_corr:       {self.relax_energies[irelax][0]-self.relax_energies[irelax][2]:15.7f} Eh')
                self.e_dsrg_mrpt3_fully_relaxed = self.relax_energies[irelax][0]
                self.e_dsrg_mrpt3 = self.e_dsrg_mrpt3_fully_relaxed

            if (len(self.state_avg) > 1):
                print_energies(np.real(self.dsrg_mrpt3_relax_eigvals_shifted[self.state_avg]))
            print(f'Time taken:                  {_t1-_t0:15.7f} s')
            print('='*47)
    
    def compute_energy_pt3_1(self):
        t0 = time.time()
        fixed_args = (self.cumulants['gamma1'], self.cumulants['eta1'], self.cumulants['lambda2'], self.cumulants['lambda3'], self)

        self.H0A1_1b = np.zeros((self.nlrg,self.nlrg), dtype='complex128')
        self.H0A1_2b = np.zeros((self.nlrg,self.nlrg,self.nlrg,self.nlrg), dtype='complex128')
        H1_T2_C2(None, self.H0A1_2b, self.F0, None, None, self.T2_1, *fixed_args)
        H1_T2_C1(self.H0A1_1b, None, self.F0, None, None, self.T2_1, *fixed_args)
        H1_T1_C1(self.H0A1_1b, None, self.F0, None, self.T1_1, None, *fixed_args)
        antisymmetrize_and_hermitize(self.H0A1_2b)
        self.H0A1_1b += self.H0A1_1b.T.conj()

        H0A1A1_1b = np.zeros((self.nlrg,self.nlrg), dtype='complex128')
        H0A1A1_2b = np.zeros((self.nlrg,self.nlrg,self.nlrg,self.nlrg), dtype='complex128')

        H1_T1_C1(H0A1A1_1b, None,      self.H0A1_1b, None,    self.T1_1, None,    *fixed_args)
        H1_T2_C1(H0A1A1_1b, None,      self.H0A1_1b, None,    None,    self.T2_1, *fixed_args)
        H2_T1_C1(H0A1A1_1b, None,      None,    self.H0A1_2b, self.T1_1, None,    *fixed_args)
        H2_T2_C1(H0A1A1_1b, None,      None,    self.H0A1_2b, None,    self.T2_1, *fixed_args)
        H1_T2_C2(None,      H0A1A1_2b, self.H0A1_1b, None,    None,    self.T2_1, *fixed_args)
        H2_T1_C2(None,      H0A1A1_2b, None,    self.H0A1_2b, self.T1_1, None,    *fixed_args)
        H2_T2_C2(None,      H0A1A1_2b, None,    self.H0A1_2b, None,    self.T2_1, *fixed_args)
        antisymmetrize_and_hermitize(H0A1A1_2b)
        H0A1A1_1b += H0A1A1_1b.T.conj()

        self.e_dsrg_mrpt3_1 = (-1./6) * H_T_C0(None, None, H0A1A1_1b, H0A1A1_2b, self.T1_1, self.T2_1, *fixed_args)

        if (self.relax_ref):
            H_T_C1_aa(self.hbar1, None, H0A1A1_1b, H0A1A1_2b, self.T1_1, self.T2_1, *fixed_args, scale=-1./12)
            H_T_C2_aaaa(None, self.hbar2, H0A1A1_1b, H0A1A1_2b, self.T1_1, self.T2_1, *fixed_args, scale=-1./12)

        del H0A1A1_1b, H0A1A1_2b

        gc.collect()
        t1 = time.time()
        if (self.verbose): print(f'... compute_energy_pt3_1: {t1-t0:15.7f} s')

    def compute_energy_pt2(self, eri):
        t0 = time.time()
        fixed_args = (self.cumulants['gamma1'], self.cumulants['eta1'], self.cumulants['lambda2'], self.cumulants['lambda3'], self)
        self.F_1_tilde = np.zeros(self.fock.shape, dtype='complex128')
        self.F_1_tilde[self.hole,self.part] = np.copy(self.fock[self.hole,self.part].conj())
        self.F_1_tilde[self.hole,self.part] += self.F_1_tilde[self.hole,self.part] * self.d1_exp
        self.F_1_tilde[self.hole,self.part] += np.multiply(self.d1_exp, np.einsum('xu,iuax,xu->ia',self.denom_act,self.T2_1[:,self.ha,:,self.pa],self.cumulants['gamma1']))
        self.F_1_tilde = np.conj(self.F_1_tilde).T

        self.V_1_tilde = np.zeros(eri.shape, dtype='complex128')
        self.V_1_tilde[self.hole,self.hole,self.part,self.part] = eri[self.hole,self.hole,self.part,self.part].copy()
        self.V_1_tilde[self.hole,self.hole,self.part,self.part] += self.V_1_tilde[self.hole,self.hole,self.part,self.part] * self.d2_exp
        self.V_1_tilde = np.einsum('ijab->abij',self.V_1_tilde)

        self.e_dsrg_mrpt2 = H_T_C0(None, None, self.F_1_tilde, self.V_1_tilde, self.T1_1, self.T2_1, *fixed_args)

        if (self.relax_ref):
            H_T_C1_aa(self.hbar1, None, self.F_1_tilde, self.V_1_tilde, self.T1_1, self.T2_1, *fixed_args, scale=0.5)
            H_T_C2_aaaa(None, self.hbar2, self.F_1_tilde, self.V_1_tilde, self.T1_1, self.T2_1, *fixed_args, scale=0.5)

        gc.collect()
        t1 = time.time()
        if (self.verbose): print(f'... compute_energy_pt2: {t1-t0:15.7f} s')

    def compute_energy_pt3_2(self, eri):
        t0 = time.time()
        fixed_args = (self.cumulants['gamma1'], self.cumulants['eta1'], self.cumulants['lambda2'], self.cumulants['lambda3'], self)
        Htilde1_1b = self.H0A1_1b + 2*self.F1.conj()
        Htilde1_2b = self.H0A1_2b + 2*eri.conj()

        self.Htilde1A1_1b = np.zeros((self.nlrg,self.nlrg), dtype='complex128')
        self.Htilde1A1_2b = np.zeros((self.nlrg,self.nlrg,self.nlrg,self.nlrg), dtype='complex128')

        H1_T1_C1(self.Htilde1A1_1b, None, Htilde1_1b, None,          self.T1_1, None,    *fixed_args)
        H1_T2_C1(self.Htilde1A1_1b, None, Htilde1_1b, None,          None,    self.T2_1, *fixed_args)
        H2_T1_C1(self.Htilde1A1_1b, None, None, Htilde1_2b, self.T1_1, None,    *fixed_args)
        H2_T2_C1(self.Htilde1A1_1b, None, None, Htilde1_2b, None,    self.T2_1, *fixed_args)
        H1_T2_C2(None, self.Htilde1A1_2b, Htilde1_1b, None,          None,    self.T2_1, *fixed_args)
        H2_T1_C2(None, self.Htilde1A1_2b, None, Htilde1_2b, self.T1_1, None,    *fixed_args)
        H2_T2_C2(None, self.Htilde1A1_2b, None, Htilde1_2b, None,    self.T2_1, *fixed_args)
        antisymmetrize_and_hermitize(self.Htilde1A1_2b)
        self.Htilde1A1_1b += self.Htilde1A1_1b.T.conj()
        self.T1_2, self.T2_2 = self.form_amplitudes(0.5*self.Htilde1A1_1b, 0.5*self.Htilde1A1_2b)
        self.e_dsrg_mrpt3_2 = H_T_C0(None, None, Htilde1_1b, Htilde1_2b, self.T1_2, self.T2_2, *fixed_args)

        if (self.relax_ref):
            H_T_C1_aa(self.hbar1, None, Htilde1_1b, Htilde1_2b, self.T1_2, self.T2_2, *fixed_args, scale=0.5)
            H_T_C2_aaaa(None, self.hbar2, Htilde1_1b, Htilde1_2b, self.T1_2, self.T2_2, *fixed_args, scale=0.5)

        del Htilde1_1b, Htilde1_2b

        gc.collect()
        t1 = time.time()
        if (self.verbose): print(f'... compute_energy_pt3_2: {t1-t0:15.7f} s')

    def compute_energy_pt3_3(self):
        t0 = time.time()
        fixed_args = (self.cumulants['gamma1'], self.cumulants['eta1'], self.cumulants['lambda2'], self.cumulants['lambda3'], self)

        H0A2_1b = np.zeros((self.nlrg,self.nlrg), dtype='complex128')
        H0A2_2b = np.zeros((self.nlrg,self.nlrg,self.nlrg,self.nlrg), dtype='complex128')
        H1_T2_C2(None, H0A2_2b, self.F0, None, None, self.T2_2, *fixed_args)
        H1_T2_C1(H0A2_1b, None, self.F0, None, None, self.T2_2, *fixed_args)
        H1_T1_C1(H0A2_1b, None, self.F0, None, self.T1_2, None, *fixed_args)
        antisymmetrize_and_hermitize(H0A2_2b)
        H0A2_1b += H0A2_1b.T.conj()

        Hbar2_1b = H0A2_1b + 0.5*self.Htilde1A1_1b
        Hbar2_2b = H0A2_2b + 0.5*self.Htilde1A1_2b

        self.e_dsrg_mrpt3_3 = H_T_C0(None, None, Hbar2_1b, Hbar2_2b, self.T1_1, self.T2_1, *fixed_args)

        if (self.relax_ref):
            H_T_C1_aa(self.hbar1, None, Hbar2_1b, Hbar2_2b, self.T1_1, self.T2_1, *fixed_args, scale=0.5)
            H_T_C2_aaaa(None, self.hbar2, Hbar2_1b, Hbar2_2b, self.T1_1, self.T2_1, *fixed_args, scale=0.5)

        del H0A2_1b, H0A2_2b, Hbar2_1b, Hbar2_2b

        gc.collect()
        t1 = time.time()
        if (self.verbose): print(f'... compute_energy_pt3_3: {t1-t0:15.7f} s')

    def renormalize_F(self, H1, T2):
        F_1_tilde = H1[self.hole,self.part].copy()
        F_1_tilde += F_1_tilde * self.d1_exp
        F_1_tilde += np.multiply(self.d1_exp, np.einsum('xu,iuax,xu->ia',self.denom_act,T2[:,self.ha,:,self.pa],self.cumulants['gamma1']))
        return F_1_tilde
    
    def renormalize_V(self, H2):
        V_1_tilde = H2[self.hole,self.hole,self.part,self.part].copy()
        V_1_tilde += V_1_tilde * self.d2_exp
        return V_1_tilde

    def dsrg_mrpt3_update(self, s, eri, pt2_only):
        if (self.relax_ref):
            self.hbar1 = np.zeros((self.nact,self.nact),dtype='complex128')
            self.hbar2 = np.zeros((self.nact,self.nact,self.nact,self.nact),dtype='complex128')

        self.cumulants = make_cumulants(self.rdms)
        self.form_denominators(s)
        self.T1_1, self.T2_1 = self.form_amplitudes(self.fock.conj(), eri.conj())

        self.e_dsrg_mrpt3_1 = self.e_dsrg_mrpt2 = self.e_dsrg_mrpt3_2 = self.e_dsrg_mrpt3_3 = .0j

        if (not pt2_only):
            self.compute_energy_pt3_1()
        self.compute_energy_pt2(eri)
        if (not pt2_only):
            self.compute_energy_pt3_2(eri)
            self.compute_energy_pt3_3()

        self.e_dsrg_mrpt3 = self.e_dsrg_mrpt3_1 + self.e_dsrg_mrpt2 + self.e_dsrg_mrpt3_2 + self.e_dsrg_mrpt3_3
        gc.collect()

    def run_dsrg_mrpt2(self, s, relativistic, relax=None, relax_convergence=1e-8, maxiter=20):
        """
        Tensor storage convention follows that of 10.1021/acs.jctc.5b00134, i.e.,
        F_a^i = <a|f|i>
        H_{ab}^{ij} = <ab|H|ij>
        gamma1_a^i = <Phi|i+ a|Phi> (!)
        """
        self.relax = relax
        if (relativistic):
            _eri = self.dhf_eri_full_asym
            _method = 'Relativistic'
            _hcore = self.dhf_hcore_mo
        else:
            _eri = self.rhf_eri_full_asym
            _method = 'Non-relativistic'
            _hcore = self.rhf_hcore_spinorb

        if (self.verbose):
            print('')
            print('='*47)
            print('{:^47}'.format(f'{_method} DSRG-MRPT2'))
            print('='*47)

        _t0 = time.time()

        if (relax is None):
            nrelax = 0
        elif (relax == 'once'):
            nrelax = 1
        elif (relax == 'twice'):
            nrelax = 2
        elif (relax == 'iterate'):
            nrelax = maxiter
        elif (type(relax) is int):
            if (relax < 0):
                raise Exception(f'Relax iteration must be positive!')
            else:
                nrelax = min(maxiter, relax)
        else:
            raise Exception(f'Relax option {relax} is not implemented yet!')
        
        _t2 = time.time()
        self.converged = False
        self.dsrg_mrpt2_update(s, _eri) # We need to do mrpt2 at least once
        _verbose = self.verbose

        if (nrelax > 0):
            self.relax_energies = np.zeros((maxiter,3)) # [iter, [unrelaxed, relaxed, Eref]]
            
            self.verbose = False
            if (_verbose):
                print('-'*47)
                print('{:^47}'.format(f'DSRG-MRPT2 reference relaxation'))
                print('-'*47)
                print('{:<30}{:<20}{:<10}'.format('Iter','Energy','Delta E'))
                print(f'   -Eref  {self.e_casci:.7f}       ')
                print(f'   -Ecorr {self.e_dsrg_mrpt2.real:.7f}       ')
        else:
            self.relax_energies = np.zeros((1,3))
            self.relax_energies[0, 0] = self.e_dsrg_mrpt2.real+self.e_casci
            self.relax_energies[0, 2] = self.e_casci
        
        for irelax in range(nrelax):
            self.relax_energies[irelax, 0] = self.e_dsrg_mrpt2.real+self.e_casci
            self.relax_energies[irelax, 2] = self.e_casci

            if (_verbose): print('{:<30}{:<20}{:<10}'.format(f'<Psi^({irelax:d})|Hbar^({irelax:d})|Psi^({irelax:d})>',f'{self.e_dsrg_mrpt2.real+self.e_casci:.7f}',f'{self.relax_energies[irelax][0]-self.relax_energies[irelax-1][1]:.5e}'))

            if (nrelax == 2 and irelax == 1): break
            self.dsrg_mrpt2_reference_relaxation(_eri)
            self.relax_energies[irelax, 1] = self.e_dsrg_mrpt2_relaxed
            if (_verbose): print(f'   -Erelax{self.e_relax:.7f}       ')
            if (_verbose): print('{:<30}{:<20}{:<10}'.format(f'<Psi^({irelax:d})|Hbar^({irelax+1:d})|Psi^({irelax:d})>',f'{self.e_dsrg_mrpt2_relaxed:.7f}',f'{self.relax_energies[irelax][1]-self.relax_energies[irelax][0]:.5e}'))
            if (self.test_relaxation_convergence(irelax, relax_convergence)): break
            if (nrelax == 1): break

            _psi = [self.dsrg_mrpt2_relax_eigvecs[:,i] for i in self.state_avg]
            self.rdms_canon['1rdm'] = get_1_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)
            self.rdms_canon['2rdm'] = get_2_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)
            self.rdms_canon['3rdm'] = get_3_rdm_sa(self.det_strings, self.cas, _psi, self.sa_weights, self.verbose)

            _eri_canon = np.einsum('ip,jq,pqrs,kr,ls->ijkl',np.conjugate(self.semicanonicalizer),np.conjugate(self.semicanonicalizer),_eri,self.semicanonicalizer,self.semicanonicalizer,optimize='optimal')
            _hcore_canon = np.einsum('ip,pq,jq->ij',np.conjugate(self.semicanonicalizer),_hcore,self.semicanonicalizer,optimize='optimal')
            del _eri, _hcore

            _gen_fock_canon = _hcore_canon.copy() + np.einsum('piqi->pq',_eri_canon[:,self.core,:,self.core]) + np.einsum('piqj,ij->pq',_eri_canon[:,self.active,:,self.active],self.rdms_canon['1rdm'])

            _gen_fock_diag = np.zeros(_gen_fock_canon.shape, dtype='complex128')
            _gen_fock_diag[self.core,self.core] = _gen_fock_canon[self.core,self.core].copy()
            _gen_fock_diag[self.active,self.active] = _gen_fock_canon[self.active,self.active].copy()
            _gen_fock_diag[self.virt,self.virt] = _gen_fock_canon[self.virt,self.virt].copy()
            _, self.semicanonicalizer = np.linalg.eigh(_gen_fock_diag)
            self.gen_fock_semicanon = np.einsum('ip,ij,jq->pq',np.conj(self.semicanonicalizer), _gen_fock_canon, self.semicanonicalizer)
            self.fock = self.gen_fock_semicanon
            self.semicanonicalizer_active = self.semicanonicalizer[self.active, self.active]

            self.rdms['1rdm'] = np.einsum('ip,ij,jq->pq', self.semicanonicalizer_active, self.rdms_canon['1rdm'], np.conj(self.semicanonicalizer_active), optimize='optimal')
            self.rdms['2rdm'] = np.einsum('ip,jq,ijkl,kr,ls->pqrs', self.semicanonicalizer_active, self.semicanonicalizer_active, self.rdms_canon['2rdm'], np.conj(self.semicanonicalizer_active),np.conj(self.semicanonicalizer_active), optimize='optimal')
            self.rdms['3rdm'] = np.einsum('ip,jq,kr,ijklmn,ls,mt,nu->pqrstu', self.semicanonicalizer_active, self.semicanonicalizer_active, self.semicanonicalizer_active, self.rdms_canon['3rdm'], np.conj(self.semicanonicalizer_active),np.conj(self.semicanonicalizer_active), np.conj(self.semicanonicalizer_active), optimize='optimal')
            _eri_semican = np.einsum('ip,jq,ijkl,kr,ls->pqrs',np.conj(self.semicanonicalizer),np.conj(self.semicanonicalizer),_eri_canon,self.semicanonicalizer,self.semicanonicalizer,optimize='optimal')
            _hcore_semican = np.einsum('ip,ij,jq->pq',np.conj(self.semicanonicalizer),_hcore_canon,self.semicanonicalizer,optimize='optimal')
            _eri = _eri_semican
            _hcore = _hcore_semican
            del _eri_semican, _hcore_semican, _eri_canon, _hcore_canon

            self.e_casci = self.nuclear_repulsion
            self.e_casci += np.einsum('mm->',_hcore[self.core,self.core])
            self.e_casci += 0.5 * np.einsum('mnmn->',_eri[self.core,self.core,self.core,self.core])
            self.e_casci += np.einsum('mumv,uv->',_eri[self.core,self.active,self.core,self.active],self.rdms['1rdm'])
            self.e_casci += np.einsum('uv,uv->',_hcore[self.active,self.active],self.rdms['1rdm'])
            self.e_casci += 0.25 * np.einsum('uvxy,uvxy->',_eri[self.active,self.active,self.active,self.active],self.rdms['2rdm'])

            self.e_casci = self.e_casci.real
            if (_verbose): print(f'   -Eref  {self.e_casci:.7f}       ')
            
            self.dsrg_mrpt2_update(s, _eri)
            if (_verbose): print(f'   -Ecorr {self.e_dsrg_mrpt2.real:.7f}       ')
            
        self.verbose = _verbose

        _t3 = time.time()

        try:
            assert(abs(self.e_dsrg_mrpt2.imag) < MACHEPS)
        except AssertionError:
            print(f'Imaginary part of DSRG-MRPT2 energy, {self.e_dsrg_mrpt2.imag} is larger than {MACHEPS}')
        
        self.e_dsrg_mrpt2 = self.e_dsrg_mrpt2.real

        _t1 = time.time()

        if (self.verbose):
            print(f'Unrelaxed DSRG-MRPT2 energy:           {self.relax_energies[0][0]:15.7f} Eh')
            print(f'Unrelaxed DSRG-MRPT2 E_corr:           {self.relax_energies[0][0]-self.relax_energies[0][2]:15.7f} Eh')
            self.e_dsrg_mrpt2_unrelaxed = self.relax_energies[0][0]
            self.e_dsrg_mrpt2 = self.e_dsrg_mrpt2_unrelaxed
            if (nrelax > 0):
                print(f'Partially relaxed DSRG-MRPT2 energy:   {self.relax_energies[0][1]:15.7f} Eh')
                print(f'Partially relaxed DSRG-MRPT2 E_corr:   {self.relax_energies[0][1]-self.relax_energies[0][2]:15.7f} Eh')
                self.e_dsrg_mrpt2_partially_relaxed = self.relax_energies[0][1]
                self.e_dsrg_mrpt2 = self.e_dsrg_mrpt2_partially_relaxed
            if (nrelax > 1):
                print(f'Relaxed DSRG-MRPT2 energy:             {self.relax_energies[1][0]:15.7f} Eh')
                print(f'Relaxed DSRG-MRPT2 E_corr:             {self.relax_energies[1][0]-self.relax_energies[1][2]:15.7f} Eh')
                self.e_dsrg_mrpt2_relaxed = self.relax_energies[1][0]
                self.e_dsrg_mrpt2 = self.e_dsrg_mrpt2_relaxed
            if (nrelax > 2):
                print(f'Fully relaxed DSRG-MRPT2 energy:       {self.relax_energies[irelax][0]:15.7f} Eh')
                print(f'Fully relaxed DSRG-MRPT2 E_corr:       {self.relax_energies[irelax][0]-self.relax_energies[irelax][2]:15.7f} Eh')
                self.e_dsrg_mrpt2_fully_relaxed = self.relax_energies[irelax][0]
                self.e_dsrg_mrpt2 = self.e_dsrg_mrpt2_fully_relaxed

            if (len(self.state_avg) > 1):
                print_energies(np.real(self.dsrg_mrpt2_relax_eigvals_shifted[self.state_avg]))
            print(f'Time taken:                  {_t1-_t0:15.7f} s')
            print('='*47)

    def test_relaxation_convergence(self, n, relax_convergence):
        """
        Test convergence for reference relaxation.
        :param n: iteration number (start from 0)
        :return: True if converged
        """
        if n == 1 and self.relax == "twice":
            self.converged = True

        if n != 0 and self.relax == "iterate":
            e_diff_u = abs(self.relax_energies[n][0] - self.relax_energies[n-1][0])
            e_diff_r = abs(self.relax_energies[n][1] - self.relax_energies[n-1][1])
            e_diff = abs(self.relax_energies[n][0] - self.relax_energies[n][1])
            if all(e < relax_convergence for e in [e_diff_u, e_diff_r, e_diff]):
                self.converged = True

        return self.converged
    
    def dsrg_mrpt2_update(self, s, _eri):
        self.cumulants = make_cumulants(self.rdms)
        self.form_denominators(s)

        self.T2_1 = np.conj(_eri[self.hole,self.hole,self.part,self.part].copy()) * self.d2
        self.T2_1[self.ha,self.ha,self.pa,self.pa] = 0

        self.T1_1 = np.conj(self.fock[self.hole,self.part].copy())
        self.T1_1 += np.einsum('xu,iuax,xu->ia', self.denom_act, self.T2_1[:,self.ha,:,self.pa],self.cumulants['gamma1'])
        self.T1_1 *= self.d1
        self.T1_1[self.ha,self.pa] = 0

        self.F_1_tilde = np.conj(self.fock[self.hole,self.part].copy())
        self.F_1_tilde += self.F_1_tilde * self.d1_exp
        self.F_1_tilde += np.multiply(self.d1_exp, np.einsum('xu,iuax,xu->ia',self.denom_act,self.T2_1[:,self.ha,:,self.pa],self.cumulants['gamma1']))
        # This conjugation is subtle: self.fock[i,a] accesses <i|f|a>, so np.conj(self.fock[i,a]) = <a|f|i> = f_a^i
        # Now F_1_tilde_a^i = f_a^i + ..., so the three lines above calculates F_1_tilde_a^i = <a|ftilde|i>, 
        # but we want to store <i|ftilde|a>, so we take the complex conjugate.
        self.F_1_tilde = np.conj(self.F_1_tilde)

        self.V_1_tilde = _eri[self.hole,self.hole,self.part,self.part].copy()
        self.V_1_tilde += self.V_1_tilde * self.d2_exp
        # V_1_tilde_ab^ij = <ab|vtilde|ij> = <ab||ij>(1+exp)
        # As we want to store <ij|vtilde|ab>, so we should properly take two complex conjugates, 
        # but since the exp part is real, we can cancel out the two conjugations.

        self.e_dsrg_mrpt2 = dsrg_HT(self.F_1_tilde, self.V_1_tilde, self.T1_1, self.T2_1, self.cumulants['gamma1'], self.cumulants['eta1'], self.cumulants['lambda2'], self.cumulants['lambda3'], self)

    def dsrg_mrpt3_reference_relaxation(self, _eri):
        self.hbar1 += self.hbar1.T.conj()
        self.hbar1 += self.fock[self.active,self.active].conj()

        antisymmetrize_and_hermitize(self.hbar2)
        self.hbar2 += _eri[self.active,self.active,self.active,self.active].conj()

        self.relax_e_scalar = -np.einsum('vu,uv->', self.hbar1, self.cumulants['gamma1']) - 0.25*np.einsum('xyuv,uvxy->',self.hbar2,self.rdms['2rdm']) + np.einsum('xyuv,ux,vy->',self.hbar2,self.cumulants['gamma1'],self.cumulants['gamma1'])

        self.hbar1 -= np.einsum('uyvx,xy->uv',self.hbar2,self.cumulants['gamma1'])

        # For now, all things to do with CASCI are in the physicist's notation
        self.hbar1 = np.conjugate(self.hbar1)
        self.hbar2 = np.conjugate(self.hbar2)

        self.hbar1_canon = np.einsum('ip,pq,jq->ij', np.conj(self.semicanonicalizer_active), self.hbar1, (self.semicanonicalizer_active), optimize='optimal')
        self.hbar2_canon = np.einsum('ip,jq,pqrs,kr,ls->ijkl', np.conj(self.semicanonicalizer_active), np.conj(self.semicanonicalizer_active), self.hbar2, (self.semicanonicalizer_active),(self.semicanonicalizer_active), optimize='optimal')

        _ref_relax_hamil = form_cas_hamiltonian(self.hbar1_canon, self.hbar2_canon, self.det_strings, self.verbose, self.cas)
        self.dsrg_mrpt3_relax_eigvals, self.dsrg_mrpt3_relax_eigvecs = np.linalg.eigh(_ref_relax_hamil)

        self.e_relax = (np.dot(self.dsrg_mrpt3_relax_eigvals[self.state_avg], self.sa_weights) + self.relax_e_scalar)
        try:
            assert(abs(self.e_relax.imag) < MACHEPS)
        except AssertionError:
            print(f'Imaginary part of DSRG-MRPT3 relaxation energy, {self.e_relax.imag} is larger than {MACHEPS}')
        self.e_relax = self.e_relax.real
        
        self.e_dsrg_mrpt3_relaxed = (self.e_casci + self.e_dsrg_mrpt3 + self.e_relax).real
        self.dsrg_mrpt3_relax_eigvals_shifted = (self.e_casci + self.e_dsrg_mrpt3 + self.dsrg_mrpt3_relax_eigvals + self.relax_e_scalar)

    def dsrg_mrpt2_reference_relaxation(self, _eri):
        _hbar2 = _eri[self.active,self.active,self.active,self.active].copy()
        _C2 = 0.5*Hbar_active_twobody_wicked(self, self.F_1_tilde, self.V_1_tilde, self.T1_1, self.T2_1, self.cumulants['gamma1'], self.cumulants['eta1'])
        # 0.5*[H, T-T+] = 0.5*([H, T] + [H, T]+)
        _hbar2 += _C2 + np.einsum('ijab->abij',np.conj(_C2)) 

        _hbar1 = self.fock[self.active,self.active].copy()
        _C1 = 0.5*Hbar_active_onebody_wicked(self, self.F_1_tilde, self.V_1_tilde, self.T1_1, self.T2_1, self.cumulants['gamma1'], self.cumulants['eta1'], self.cumulants['lambda2'])
        # 0.5*[H, T-T+] = 0.5*([H, T] + [H, T]+)
        _hbar1 += _C1 + np.einsum('ia->ai',np.conj(_C1))

        _e_scalar = -np.einsum('uv,uv->', _hbar1, self.cumulants['gamma1']) - 0.25*np.einsum('uvxy,uvxy->',_hbar2,self.rdms['2rdm']) + np.einsum('uvxy,ux,vy->',_hbar2,self.cumulants['gamma1'],self.cumulants['gamma1'])

        _hbar1 -= np.einsum('uxvy,xy->uv',_hbar2,self.cumulants['gamma1'])

        if (self.semi_can):
            _hbar1_canon = np.einsum('ip,pq,jq->ij', np.conj(self.semicanonicalizer_active), _hbar1, (self.semicanonicalizer_active), optimize='optimal')
            _hbar2_canon = np.einsum('ip,jq,pqrs,kr,ls->ijkl', np.conj(self.semicanonicalizer_active), np.conj(self.semicanonicalizer_active), _hbar2, (self.semicanonicalizer_active),(self.semicanonicalizer_active), optimize='optimal')
        else:
            _hbar1_canon = _hbar1
            _hbar2_canon = _hbar2

        _ref_relax_hamil = form_cas_hamiltonian(_hbar1_canon, _hbar2_canon, self.det_strings, self.verbose, self.cas)
        self.dsrg_mrpt2_relax_eigvals, self.dsrg_mrpt2_relax_eigvecs = np.linalg.eigh(_ref_relax_hamil)

        self.e_relax = (np.dot(self.dsrg_mrpt2_relax_eigvals[self.state_avg], self.sa_weights) + _e_scalar)
        try:
            assert(abs(self.e_relax.imag) < MACHEPS)
        except AssertionError:
            print(f'Imaginary part of DSRG-MRPT2 relaxation energy, {self.e_relax.imag} is larger than {MACHEPS}')
        self.e_relax = self.e_relax.real
        
        self.e_dsrg_mrpt2_relaxed = (self.e_casci + self.e_dsrg_mrpt2 + self.e_relax).real
        self.dsrg_mrpt2_relax_eigvals_shifted = (self.e_casci + self.e_dsrg_mrpt2 + self.dsrg_mrpt2_relax_eigvals + _e_scalar)

    def read_in_mo(self, relativistic, mo_coeff_in, frozen=None, debug=False, algo='direct', erifile=None, eriread=None):
        if type(mo_coeff_in) is str:
            _mo_coeff = np.load(mo_coeff_in)
        else:
            _mo_coeff = mo_coeff_in
        if (not relativistic):
            self.nonrel_ao2mo(_mo_coeff, frozen)
        else:
            self.rel_ao2mo(_mo_coeff, frozen, algo, erifile, eriread)

    def nonrel_ao2mo(self, mo_coeff, frozen):
        _t0 = time.time()
        print('Building integrals...')
        
        _rhf_hcore_ao = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc')
        _rhf_hcore_mo = np.einsum('pi,pq,qj->ij', mo_coeff, _rhf_hcore_ao, mo_coeff)

        _rhf_hcore_spinorb = np.zeros((self.nlrg,self.nlrg),dtype='complex128') # Interleaved 1a1b 2a2b 3a3b ....
        _rhf_hcore_spinorb[::2,::2] = _rhf_hcore_spinorb[1::2,1::2] = _rhf_hcore_mo

        if (frozen == 0 or frozen == (0,0)):
            frozen = None

        if (frozen is not None):
            print('{:#^47}'.format('Freezing orbitals!')) 
            try:
                assert (type(frozen) is int or type(frozen) is tuple)
                if (type(frozen) is int):
                    assert (frozen >= 0 and frozen < self.nelec)
                else:
                    assert ((frozen[0] >= 0 and frozen[0] < self.nelec) and frozen[1] <= self.nvirtual)
            except AssertionError:
                raise Exception("The 'frozen' argument must be an integer or tuple of integers, and they have to fit into the spectrum!")
            
            if (type(frozen) is int):
                self.nfrozen = frozen
                self.nfrozen_virt = 0
            else:
                self.nfrozen = frozen[0]
                self.nfrozen_virt = frozen[1]
            _nlrg = self.nlrg # We need to preserve the original nlrg for just a little bit longer..
            self.nlrg -= (self.nfrozen + self.nfrozen_virt)
            
            self.nelec -= self.nfrozen
            self.nocc -= self.nfrozen
            self.nvirtual -= self.nfrozen_virt

            _frzc = slice(0,self.nfrozen)
            _actv = slice(self.nfrozen,self.nlrg+self.nfrozen)

            self.e_frozen = np.einsum('ii->',_rhf_hcore_spinorb[_frzc,_frzc]) # The 1e part is common to both with DF and without, the 2e part is done later
        else:
            self.nfrozen = 0
            self.e_frozen = 0.0
            self.rhf_hcore_spinorb = _rhf_hcore_spinorb
            _frzc = slice(0,self.nfrozen)
            _actv = slice(self.nfrozen,self.nlrg+self.nfrozen)
            _actv_spatorb = slice(int(self.nfrozen/2),self.nao+int(self.nfrozen/2))
            _nlrg = self.nlrg

        # [todo] - do ERI transformation with L tensors
        if (self.density_fitting):
            self.naux = self.rhf.with_df._cderi.shape[0]
            _mem = 2*(self.naux*(self.mol.nao)**2)*16/1e9
            if (_mem < 1.0):
                if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the DF AO ERI tensor!')
            else:
                if (self.verbose): print(f'Will now allocate {_mem:.3f} GB memory for the DF AO ERI tensor!')

            _Lpq = pyscf.lib.unpack_tril(self.rhf.with_df._cderi)
            _rhf_eri_ao = np.einsum('Lpq,Lrs->pqrs', _Lpq, _Lpq)
        else:
            _rhf_eri_ao = self.mol.intor('int2e_sph') # Chemist's notation (ij|kl)

        _rhf_eri_mo = pyscf.ao2mo.incore.full(_rhf_eri_ao, mo_coeff)
        _rhf_eri_spatorb = _rhf_eri_mo.swapaxes(1,2)

        _rhf_eri_full_asym = np.zeros((_nlrg,_nlrg,_nlrg,_nlrg),dtype='complex128') # Interleaved 1a1b 2a2b 3a3b ....
        _rhf_eri_full_asym[::2,::2,::2,::2] = _rhf_eri_full_asym[1::2,1::2,1::2,1::2] = _rhf_eri_spatorb - _rhf_eri_spatorb.swapaxes(2,3) # <aa||aa> and <bb||bb>
        _rhf_eri_full_asym[::2,1::2,::2,1::2] = _rhf_eri_full_asym[1::2,::2,1::2,::2] = _rhf_eri_spatorb # <ab||ab> = <ba||ba> = <ab|ab>
        _rhf_eri_full_asym[::2,1::2,1::2,::2] = _rhf_eri_full_asym[1::2,::2,::2,1::2] = -_rhf_eri_spatorb.swapaxes(2,3) # <ab||ba> = <ba||ab> = -<ab|ab>

        if (frozen is not None):
            self.e_frozen += 0.5*np.einsum('ijij->',_rhf_eri_full_asym[:self.nfrozen,:self.nfrozen,:self.nfrozen,:self.nfrozen])

            self.rhf_hcore_spinorb = _rhf_hcore_spinorb[_actv, _actv].copy() + np.einsum('ipjp->ij',_rhf_eri_full_asym[_actv,_frzc,_actv,_frzc])
            self.rhf_eri_full_asym = _rhf_eri_full_asym[_actv,_actv,_actv,_actv]
            del _rhf_eri_full_asym
        else:
            self.e_frozen = 0.0
            self.rhf_eri_full_asym = _rhf_eri_full_asym
            self.rhf_hcore_spinorb = _rhf_hcore_spinorb

        self.nuclear_repulsion += self.e_frozen

        if (self.verbose):
            _t1 = time.time()
            print(f'Integral build time:         {_t1-_t0:15.7f} s')
            print('-'*47)

    def rel_ao2mo_einsum(self, mo_coeff, nlrg, eri, moslice):
        _mo_l = mo_coeff[:nlrg, nlrg:]
        _mo_s = mo_coeff[nlrg:, nlrg:]

        # LLLL
        eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_l[:,moslice[0]]),(_mo_l[:,moslice[1]]),self.mol.intor('int2e_spinor'),np.conj(_mo_l[:,moslice[2]]),(_mo_l[:,moslice[3]]),optimize=True)
        # SSSS
        eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_s[:,moslice[0]]),(_mo_s[:,moslice[1]]),self.mol.intor('int2e_spsp1spsp2_spinor'),np.conj(_mo_s[:,moslice[2]]),(_mo_s[:,moslice[3]]),optimize=True)/(2*self.c0)**4
        # SSLL
        eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_s[:,moslice[0]]),(_mo_s[:,moslice[1]]),self.mol.intor('int2e_spsp1_spinor'),np.conj(_mo_l[:,moslice[2]]),(_mo_l[:,moslice[3]]),optimize=True)/(2*self.c0)**2
        # LLSS
        eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_l[:,moslice[0]]),(_mo_l[:,moslice[1]]),self.mol.intor('int2e_spsp2_spinor'),np.conj(_mo_s[:,moslice[2]]),(_mo_s[:,moslice[3]]),optimize=True)/(2*self.c0)**2

        if (self.dhf.with_breit):
            # LSLS
            eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_l[:,moslice[0]]),(_mo_s[:,moslice[1]]),self.mol.intor('int2e_breit_ssp1ssp2_spinor'),np.conj(_mo_l[:,moslice[2]]),(_mo_s[:,moslice[3]]),optimize=True)/(2*self.c0)**2
            # LSSL
            eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_l[:,moslice[0]]),(_mo_s[:,moslice[1]]),self.mol.intor('int2e_breit_ssp1sps2_spinor'),np.conj(_mo_s[:,moslice[2]]),(_mo_l[:,moslice[3]]),optimize=True)/(2*self.c0)**2
            # SLLS
            eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_s[:,moslice[0]]),(_mo_l[:,moslice[1]]),self.mol.intor('int2e_breit_sps1ssp2_spinor'),np.conj(_mo_l[:,moslice[2]]),(_mo_s[:,moslice[3]]),optimize=True)/(2*self.c0)**2
            # SLSL
            eri += np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_s[:,moslice[0]]),(_mo_l[:,moslice[1]]),self.mol.intor('int2e_breit_sps1sps2_spinor'),np.conj(_mo_s[:,moslice[2]]),(_mo_l[:,moslice[3]]),optimize=True)/(2*self.c0)**2                        
        elif (self.dhf.with_gaunt):
            # Gaunt term doesn't account for the negative sign, whereas the Breit one does
            # LSLS
            eri -= np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_l[:,moslice[0]]),(_mo_s[:,moslice[1]]),self.mol.intor('int2e_ssp1ssp2_spinor'),np.conj(_mo_l[:,moslice[2]]),(_mo_s[:,moslice[3]]),optimize=True)/(2*self.c0)**2
            # LSSL
            eri -= np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_l[:,moslice[0]]),(_mo_s[:,moslice[1]]),self.mol.intor('int2e_ssp1sps2_spinor'),np.conj(_mo_s[:,moslice[2]]),(_mo_l[:,moslice[3]]),optimize=True)/(2*self.c0)**2
            # SLLS
            eri -= np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_s[:,moslice[0]]),(_mo_l[:,moslice[1]]),self.mol.intor('int2e_sps1ssp2_spinor'),np.conj(_mo_l[:,moslice[2]]),(_mo_s[:,moslice[3]]),optimize=True)/(2*self.c0)**2
            # SLSL
            eri -= np.einsum('pi,qj,pqrs,rk,sl->ijkl', np.conj(_mo_s[:,moslice[0]]),(_mo_l[:,moslice[1]]),self.mol.intor('int2e_sps1sps2_spinor'),np.conj(_mo_s[:,moslice[2]]),(_mo_l[:,moslice[3]]),optimize=True)/(2*self.c0)**2                        
        return eri

    def rel_read_eri(self, eriread):
        _t0 = time.time()
        with np.load(eriread) as npzfile:
            self.e_frozen = npzfile['e_frozen']
            self.dhf_hcore_mo = npzfile['dhf_hcore_mo']
            self.dhf_eri_full_asym = npzfile['dhf_eri_full_asym']

        self.nuclear_repulsion += self.e_frozen
        _t1 = time.time()
        if (self.verbose):
            print(f'\nTiming report')
            print(f'....integral reading: {(_t1-_t0):15.7f} s')

    def rel_ao2mo(self, mo_coeff, frozen, algo='direct', erifile=None, eriread=None):
        def h5_write(fname, slices):
            pyscf.ao2mo.r_outcore.general(self.mol, (_mo_l[:,slices[0]], _mo_l[:,slices[1]], _mo_l[:,slices[2]], _mo_l[:,slices[3]]), erifile=fname, dataname='mo_eri', intor='int2e_spinor',aosym='s1')
            eri_h5_write(self.mol, (_mo_s[:,slices[0]],_mo_s[:,slices[1]],_mo_s[:,slices[2]],_mo_s[:,slices[3]]), 'int2e_spsp1spsp2_spinor', erifile=fname)
            eri_h5_write(self.mol, (_mo_s[:,slices[0]],_mo_s[:,slices[1]],_mo_l[:,slices[2]],_mo_l[:,slices[3]]), 'int2e_spsp1_spinor', erifile=fname)
            eri_h5_write(self.mol, (_mo_l[:,slices[0]],_mo_l[:,slices[1]],_mo_s[:,slices[2]],_mo_s[:,slices[3]]), 'int2e_spsp2_spinor', erifile=fname, terminal=(not (self.dhf.with_breit or self.dhf.with_gaunt)))
            if (self.dhf.with_breit or self.dhf.with_gaunt):
                if (self.dhf.with_gaunt):
                    eri_h5_write(self.mol, (_mo_l[:,slices[0]],_mo_s[:,slices[1]],_mo_l[:,slices[2]],_mo_s[:,slices[3]]), 'int2e_ssp1ssp2_spinor', erifile=fname, gaunt=True)
                    eri_h5_write(self.mol, (_mo_l[:,slices[0]],_mo_s[:,slices[1]],_mo_s[:,slices[2]],_mo_l[:,slices[3]]), 'int2e_ssp1sps2_spinor', erifile=fname, gaunt=True)
                    eri_h5_write(self.mol, (_mo_s[:,slices[0]],_mo_l[:,slices[1]],_mo_l[:,slices[2]],_mo_s[:,slices[3]]), 'int2e_sps1ssp2_spinor', erifile=fname, gaunt=True)
                    eri_h5_write(self.mol, (_mo_s[:,slices[0]],_mo_l[:,slices[1]],_mo_s[:,slices[2]],_mo_l[:,slices[3]]), 'int2e_sps1sps2_spinor', erifile=fname, gaunt=True, terminal=True)
                else:
                    eri_h5_write(self.mol, (_mo_l[:,slices[0]],_mo_s[:,slices[1]],_mo_l[:,slices[2]],_mo_s[:,slices[3]]), 'int2e_breit_ssp1ssp2_spinor', erifile=fname)
                    eri_h5_write(self.mol, (_mo_l[:,slices[0]],_mo_s[:,slices[1]],_mo_s[:,slices[2]],_mo_l[:,slices[3]]), 'int2e_breit_ssp1sps2_spinor', erifile=fname)
                    eri_h5_write(self.mol, (_mo_s[:,slices[0]],_mo_l[:,slices[1]],_mo_l[:,slices[2]],_mo_s[:,slices[3]]), 'int2e_breit_sps1ssp2_spinor', erifile=fname)
                    eri_h5_write(self.mol, (_mo_s[:,slices[0]],_mo_l[:,slices[1]],_mo_s[:,slices[2]],_mo_l[:,slices[3]]), 'int2e_breit_sps1sps2_spinor', erifile=fname, terminal=True)
                    
        _t0 = time.time()
        self.norb = self.mol.nao_2c()*2

        # Harvest h from DHF (Includes both S & L blocks)
        _dhf_hcore_ao = self.dhf.get_hcore()
        _dhf_hcore_mo = np.einsum('pi,pq,qj->ij', np.conj(mo_coeff[:,self.nlrg:]), _dhf_hcore_ao, mo_coeff[:,self.nlrg:])

        del _dhf_hcore_ao

        if (frozen == 0 or frozen == (0,0)):
            frozen = None

        if (frozen is not None):
            try:
                assert (type(frozen) is int or type(frozen) is tuple) 
                if (type(frozen) is int):
                    assert (frozen >= 0 and frozen < self.nelec)
                else:
                    assert ((frozen[0] >= 0 and frozen[0] < self.nelec) and frozen[1] <= self.nvirtual)
            except AssertionError:
                raise Exception("The 'frozen' argument must be an integer or tuple of integers, and they have to fit into the spectrum!")
            
            
            if (type(frozen) is int):
                self.nfrozen = frozen
                self.nfrozen_virt = 0
            else:
                self.nfrozen = frozen[0]
                self.nfrozen_virt = frozen[1]
            _nlrg = self.nlrg # We need to preserve the original nlrg for just a little bit longer..
            self.nlrg -= (self.nfrozen + self.nfrozen_virt)
            
            self.nelec -= self.nfrozen
            self.nocc -= self.nfrozen
            self.nvirtual -= self.nfrozen_virt

            _frzc = slice(0,self.nfrozen)
            _actv = slice(self.nfrozen,self.nlrg+self.nfrozen)

            self.e_frozen = np.einsum('ii->',_dhf_hcore_mo[_frzc,_frzc]) # The 1e part is common to both with DF and without, the 2e part is done later
        else:
            self.nfrozen = 0
            self.e_frozen = 0.0
            self.dhf_hcore_mo = _dhf_hcore_mo
            _frzc = slice(0,self.nfrozen)
            _actv = slice(self.nfrozen,self.nlrg+self.nfrozen)
            _nlrg = self.nlrg

        if (self.density_fitting):
            self.naux = self.dhf.with_df._cderi[0].shape[0]
            _mem = 2*(self.naux*_nlrg**2)*16/1e9
            if (_mem < 1.0):
                if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the DF AO ERI tensor!')
            else:
                if (self.verbose): print(f'Will now allocate {_mem:.3f} GB memory for the DF AO ERI tensor!')

            _Lpq_LL = pyscf.lib.unpack_tril(self.dhf.with_df._cderi[0]) # 0 is eri_LL, 1 is eri_SS
            _Lpq_SS = pyscf.lib.unpack_tril(self.dhf.with_df._cderi[1])/(2*pyscf.lib.param.LIGHT_SPEED)**2 # 0 is eri_LL, 1 is eri_SS

            _Lpq_mo_LL = np.einsum('ip,jq,lij->lpq',np.conj(mo_coeff[:_nlrg,_nlrg:]),mo_coeff[:_nlrg,_nlrg:],_Lpq_LL,optimize='optimal')
            _Lpq_mo_SS = np.einsum('ip,jq,lij->lpq',np.conj(mo_coeff[_nlrg:,_nlrg:]),mo_coeff[_nlrg:,_nlrg:],_Lpq_SS,optimize='optimal')
            del _Lpq_LL, _Lpq_SS

            if (frozen is not None):
                # The 2e part of e_frozen
                self.e_frozen += 0.5*(np.einsum('lii,ljj->',_Lpq_mo_LL[:,_frzc,_frzc],_Lpq_mo_LL[:,_frzc,_frzc])+np.einsum('lii,ljj->',_Lpq_mo_SS[:,_frzc,_frzc],_Lpq_mo_SS[:,_frzc,_frzc])+np.einsum('lii,ljj->',_Lpq_mo_SS[:,_frzc,_frzc],_Lpq_mo_LL[:,_frzc,_frzc])+np.einsum('lii,ljj->',_Lpq_mo_LL[:,_frzc,_frzc],_Lpq_mo_SS[:,_frzc,_frzc]))
                self.e_frozen -= 0.5*(np.einsum('lij,lji->',_Lpq_mo_LL[:,_frzc,_frzc],_Lpq_mo_LL[:,_frzc,_frzc])+np.einsum('lij,lji->',_Lpq_mo_SS[:,_frzc,_frzc],_Lpq_mo_SS[:,_frzc,_frzc])+np.einsum('lij,lji->',_Lpq_mo_SS[:,_frzc,_frzc],_Lpq_mo_LL[:,_frzc,_frzc])+np.einsum('lij,lji->',_Lpq_mo_LL[:,_frzc,_frzc],_Lpq_mo_SS[:,_frzc,_frzc]))
                
                self.dhf_hcore_mo = _dhf_hcore_mo[_actv,_actv].copy()
                del _dhf_hcore_mo
                self.dhf_hcore_mo += np.einsum('lpq,lii->pq',_Lpq_mo_LL[:,_actv,_actv],_Lpq_mo_LL[:,:self.nfrozen,:self.nfrozen])+np.einsum('lpq,lii->pq',_Lpq_mo_SS[:,_actv,_actv],_Lpq_mo_SS[:,:self.nfrozen,:self.nfrozen])+np.einsum('lpq,lii->pq',_Lpq_mo_SS[:,_actv,_actv],_Lpq_mo_LL[:,:self.nfrozen,:self.nfrozen])+np.einsum('lpq,lii->pq',_Lpq_mo_LL[:,_actv,_actv],_Lpq_mo_SS[:,:self.nfrozen,:self.nfrozen])
                self.dhf_hcore_mo -= np.einsum('lpi,liq->pq',_Lpq_mo_LL[:,_actv,_frzc],_Lpq_mo_LL[:,_frzc,_actv])+np.einsum('lpi,liq->pq',_Lpq_mo_SS[:,_actv,_frzc],_Lpq_mo_SS[:,_frzc,_actv])+np.einsum('lpi,liq->pq',_Lpq_mo_SS[:,_actv,_frzc],_Lpq_mo_LL[:,_frzc,_actv])+np.einsum('lpi,liq->pq',_Lpq_mo_LL[:,_actv,_frzc],_Lpq_mo_SS[:,_frzc,_actv])


            _mem = (self.nlrg**4)*16/1e9
            if (_mem < 1.0):
                if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the MO ERI tensor!')
            else:
                if (self.verbose): print(f'Will now allocate {_mem:.3f} GB memory for the MO ERI tensor!')

            self.dhf_eri_full_asym = np.einsum('lpq,lrs->pqrs',_Lpq_mo_LL[:,_actv,_actv],_Lpq_mo_LL[:,_actv,_actv],optimize='optimal') + np.einsum('lpq,lrs->pqrs',_Lpq_mo_SS[:,_actv,_actv],_Lpq_mo_SS[:,_actv,_actv],optimize='optimal') + np.einsum('lpq,lrs->pqrs',_Lpq_mo_LL[:,_actv,_actv],_Lpq_mo_SS[:,_actv,_actv],optimize='optimal') + np.einsum('lpq,lrs->pqrs',_Lpq_mo_SS[:,_actv,_actv],_Lpq_mo_LL[:,_actv,_actv],optimize='optimal')
            del _Lpq_mo_LL, _Lpq_mo_SS
            self.dhf_eri_full_asym = self.dhf_eri_full_asym.swapaxes(1,2) - self.dhf_eri_full_asym.swapaxes(1,2).swapaxes(2,3)
        else:
            if (algo == 'direct'):
                _write_eri = False
                if (type(erifile) is str):
                    _write_eri = True
                
                if (type(eriread) is str):
                    self.rel_read_eri(eriread)
                    return

                if (frozen is not None):
                    _mem = self.nfrozen**4*16/1e9*2
                    if (_mem < 1.0):
                        if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the frozen core ERI tensor!')
                    else:
                        if (self.verbose): print(f'Will now allocate {_mem:.3f} GB memory for the frozen core ERI tensor!')

                    eri_fc_ao = np.zeros((self.nfrozen,self.nfrozen,self.nfrozen,self.nfrozen),dtype='complex128')
                    eri_fc = self.rel_ao2mo_einsum(mo_coeff, _nlrg, eri_fc_ao, (_frzc,_frzc,_frzc,_frzc))
                    self.e_frozen += 0.5*(np.einsum('iijj->',eri_fc)-np.einsum('ijji->',eri_fc))
                    del eri_fc, eri_fc_ao

                    _mem = 2*(self.nlrg*self.nfrozen)**2*16/1e9*2
                    if (_mem < 1.0):
                        if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the aacc and acca ERI tensor!')
                    else:
                        if (self.verbose): print(f'Will now allocate {_mem:.3f} GB memory for the aacc and acca ERI tensor!')

                    eri_aacc_ao = np.zeros((self.nlrg,self.nlrg,self.nfrozen,self.nfrozen),dtype='complex128')
                    eri_aacc = self.rel_ao2mo_einsum(mo_coeff, _nlrg, eri_aacc_ao, (_actv,_actv,_frzc,_frzc))
                    self.dhf_hcore_mo = _dhf_hcore_mo[_actv,_actv].copy()
                    self.dhf_hcore_mo += np.einsum('pqii->pq', eri_aacc)
                    del eri_aacc_ao, eri_aacc

                    eri_acca_ao = np.zeros((self.nlrg,self.nfrozen,self.nfrozen,self.nlrg),dtype='complex128')
                    eri_acca = self.rel_ao2mo_einsum(mo_coeff, _nlrg, eri_acca_ao, (_actv,_frzc,_frzc,_actv))
                    self.dhf_hcore_mo -= np.einsum('piiq->pq', eri_acca)
                    del eri_acca_ao, eri_acca

                _mem = _nlrg**4*16/1e9
                if (_mem < 1.0):
                    if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the AO ERI tensor!')
                else:
                    if (self.verbose): print(f'Will now allocate {_mem:.3f} GB memory for the AO ERI tensor!')

                _dhf_eri_mo_full = np.zeros((self.nlrg,self.nlrg,self.nlrg,self.nlrg),dtype='complex128')
                _dhf_eri_mo_full = self.rel_ao2mo_einsum(mo_coeff, _nlrg, _dhf_eri_mo_full, (_actv,_actv,_actv,_actv))
                
                _dhf_eri_full_asym = _dhf_eri_mo_full.swapaxes(1,2) - _dhf_eri_mo_full.swapaxes(1,2).swapaxes(2,3)
        
                del _dhf_eri_mo_full

                if (frozen is not None):
                    self.dhf_eri_full_asym = _dhf_eri_full_asym
                    del _dhf_eri_full_asym
                else:
                    self.e_frozen = 0.0
                    self.dhf_eri_full_asym = _dhf_eri_full_asym
                    self.dhf_hcore_mo = _dhf_hcore_mo

                if (_write_eri):
                    np.savez(erifile, e_frozen=self.e_frozen, dhf_hcore_mo=self.dhf_hcore_mo, dhf_eri_full_asym=self.dhf_eri_full_asym)
            
            elif (algo == 'naive'):
                warnings.warn("The naive algorithm uses excessive memory and is only for testing! Please switch to the 'direct' algorithm for production runs!")
                _mem = self.norb**4*16/1e9
                if (_mem < 1.0):
                    if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB memory for the AO ERI tensor!')
                else:
                    if (self.verbose): print(f'Will now allocate {_mem:.3f} GB memory for the AO ERI tensor!')

                _dhf_eri_ao = np.zeros((self.norb,self.norb,self.norb,self.norb),dtype='complex128')
                _dhf_eri_ao[:_nlrg,:_nlrg,:_nlrg,:_nlrg] = self.mol.intor('int2e_spinor')
                _dhf_eri_ao[_nlrg:,_nlrg:,_nlrg:,_nlrg:] = self.mol.intor('int2e_spsp1spsp2_spinor')/(2*self.c0)**4
                _dhf_eri_ao[_nlrg:,_nlrg:,:_nlrg,:_nlrg] = self.mol.intor('int2e_spsp1_spinor')/(2*self.c0)**2
                _dhf_eri_ao[:_nlrg,:_nlrg,_nlrg:,_nlrg:] = self.mol.intor('int2e_spsp2_spinor')/(2*self.c0)**2
                if (self.dhf.with_breit or self.dhf.with_gaunt):
                    if (self.dhf.with_breit):
                        _dhf_eri_ao[:_nlrg,_nlrg:,:_nlrg,_nlrg:] = self.mol.intor('int2e_breit_ssp1ssp2_spinor')/(2*self.c0)**2
                        _dhf_eri_ao[:_nlrg,_nlrg:,_nlrg:,:_nlrg] = self.mol.intor('int2e_breit_ssp1sps2_spinor')/(2*self.c0)**2
                        _dhf_eri_ao[_nlrg:,:_nlrg,:_nlrg,_nlrg:] = self.mol.intor('int2e_breit_sps1ssp2_spinor')/(2*self.c0)**2
                        _dhf_eri_ao[_nlrg:,:_nlrg,_nlrg:,:_nlrg] = self.mol.intor('int2e_breit_sps1sps2_spinor')/(2*self.c0)**2
                    else:
                        # Gaunt term doesn't account for the negative sign, whereas the Breit one does
                        _dhf_eri_ao[:_nlrg,_nlrg:,:_nlrg,_nlrg:] = -self.mol.intor('int2e_ssp1ssp2_spinor')/(2*self.c0)**2
                        _dhf_eri_ao[:_nlrg,_nlrg:,_nlrg:,:_nlrg] = -self.mol.intor('int2e_ssp1sps2_spinor')/(2*self.c0)**2
                        _dhf_eri_ao[_nlrg:,:_nlrg,:_nlrg,_nlrg:] = -self.mol.intor('int2e_sps1ssp2_spinor')/(2*self.c0)**2
                        _dhf_eri_ao[_nlrg:,:_nlrg,_nlrg:,:_nlrg] = -self.mol.intor('int2e_sps1sps2_spinor')/(2*self.c0)**2
                
                _dhf_eri_mo_full = np.einsum('pi,qj,pqrs,rk,sl->ijkl',np.conj(mo_coeff[:,_nlrg:]),(mo_coeff[:,_nlrg:]),_dhf_eri_ao,np.conj(mo_coeff[:,_nlrg:]),(mo_coeff[:,_nlrg:]),optimize=True)
                _dhf_eri_full_asym = _dhf_eri_mo_full.swapaxes(1,2) - _dhf_eri_mo_full.swapaxes(1,2).swapaxes(2,3)
        
                del _dhf_eri_ao, _dhf_eri_mo_full

                if (frozen is not None):
                    self.e_frozen += 0.5*np.einsum('ijij->',_dhf_eri_full_asym[:self.nfrozen,:self.nfrozen,:self.nfrozen,:self.nfrozen])

                    self.dhf_hcore_mo = _dhf_hcore_mo[_actv, _actv].copy() + np.einsum('ipjp->ij',_dhf_eri_full_asym[_actv,_frzc,_actv,_frzc])
                    del _dhf_hcore_mo
                    self.dhf_eri_full_asym = _dhf_eri_full_asym[_actv,_actv,_actv,_actv]
                    del _dhf_eri_full_asym
                else:
                    self.e_frozen = 0.0
                    self.dhf_eri_full_asym = _dhf_eri_full_asym
                    self.dhf_hcore_mo = _dhf_hcore_mo
            elif (algo == 'disk'):
                _cleanup = True
                if (erifile is None or type(erifile) is not str):
                    fname = 'tmp'
                else:
                    fname = erifile
                    _cleanup = False
                
                _read_eri = False
                if (type(eriread) is str):
                    fname = eriread
                    _read_eri = True
                    _cleanup = False

                _fname = fname

                _mo_l = mo_coeff[:_nlrg, _nlrg:]
                _mo_s = mo_coeff[_nlrg:, _nlrg:]/(2*self.c0)

                if (frozen is not None):
                    _mem = self.nfrozen**4*16/1e9*2
                    if (_mem < 1.0):
                        if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB disk space for the frozen core ERI tensor!')
                    else:
                        if (self.verbose): print(f'Will now allocate {_mem:.3f} GB disk space for the frozen core ERI tensor!')

                    fname = _fname + 'cccc.h5'
                    if (not _read_eri): h5_write(fname, (_frzc,_frzc,_frzc,_frzc))
                    with h5py.File(_fname+'cccc.h5', mode='r') as feri:
                        eri_fc = feri['mo_eri'][:].reshape((self.nfrozen, self.nfrozen, self.nfrozen, self.nfrozen))
                    self.e_frozen += 0.5*(np.einsum('iijj->',eri_fc)-np.einsum('ijji->',eri_fc))
                    if (_cleanup): os.remove(fname)
                    del eri_fc

                    _mem = 2*(self.nlrg*self.nfrozen)**2*16/1e9*2
                    if (_mem < 1.0):
                        if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB disk space for the aacc and acca ERI tensor!')
                    else:
                        if (self.verbose): print(f'Will now allocate {_mem:.3f} GB disk space for the aacc and acca ERI tensor!')

                    fname = _fname + 'aacc.h5'
                    if (not _read_eri): h5_write(fname, (_actv,_actv,_frzc,_frzc))
                    with h5py.File(_fname+'aacc.h5', mode='r') as feri:
                        eri_fc_pqii = feri['mo_eri'][:].reshape((self.nlrg, self.nlrg, self.nfrozen, self.nfrozen))
                    self.dhf_hcore_mo = _dhf_hcore_mo[_actv,_actv].copy()
                    self.dhf_hcore_mo += np.einsum('pqii->pq', eri_fc_pqii)
                    if (_cleanup): os.remove(fname)
                    del eri_fc_pqii

                    fname = _fname + 'acca.h5'
                    if (not _read_eri): h5_write(fname, (_actv,_frzc,_frzc,_actv))
                    with h5py.File(_fname+'acca.h5', mode='r') as feri:
                        eri_fc_piiq = feri['mo_eri'][:].reshape((self.nlrg, self.nfrozen, self.nfrozen, self.nlrg))
                    self.dhf_hcore_mo -= np.einsum('piiq->pq', eri_fc_piiq)
                    if (_cleanup): os.remove(fname)
                    del eri_fc_piiq

                _mem = self.nlrg**4*16/1e9*2
                if (_mem < 1.0):
                    if (self.verbose): print(f'Will now allocate {_mem*1000:.3f} MB disk space for the AO ERI tensor!')
                else:
                    if (self.verbose): print(f'Will now allocate {_mem:.3f} GB disk space for the AO ERI tensor!')

                fname = _fname + '.h5'
                if (not _read_eri): h5_write(fname, (_actv,_actv,_actv,_actv))
                with h5py.File(fname, 'r') as feri:
                    _dhf_eri_mo_full = feri['mo_eri'][:].reshape((self.nlrg,self.nlrg,self.nlrg,self.nlrg))
                if (_cleanup): os.remove(fname)
                    
                _dhf_eri_full_asym = _dhf_eri_mo_full.swapaxes(1,2) - _dhf_eri_mo_full.swapaxes(1,2).swapaxes(2,3)

                if (frozen is not None):
                    self.dhf_eri_full_asym = _dhf_eri_full_asym
                    del _dhf_eri_full_asym, _dhf_eri_mo_full
                else:
                    self.e_frozen = 0.0
                    self.dhf_eri_full_asym = _dhf_eri_full_asym
                    self.dhf_hcore_mo = _dhf_hcore_mo
            
        self.nuclear_repulsion += self.e_frozen
        _t1 = time.time()
        if (self.verbose):
            print(f'\nTiming report')
            print(f'....integral transformation: {(_t1-_t0):15.7f} s')

class ACISolver:
    def __init__(self, sys, pspace0, verbose, cas, sigma, gamma, relativistic, maxiter=50, pt2=True, etol=1e-8):
        self.p_space_det_strings = np.array(pspace0)
        self.ndets = len(pspace0)
        self.verbose = verbose
        self.cas = cas
        self.sys = sys # A RelForte object
        self.iter = 1
        self.sigma = sigma
        self.gamma = gamma
        self.maxiter = maxiter
        self.pt2 = pt2
        self.relativistic = relativistic
        self.etol = etol
        
        if (relativistic):
            self.H1body = self.sys.dhf_hcore_mo
            self.H2body = self.sys.dhf_eri_full_asym
        else:
            self.H1body = self.sys.rhf_hcore_spinorb
            self.H2body = self.sys.rhf_eri_full_asym
        
        self.ncas_elec, self.ncas_orbs = cas
        self.ncoreel = self.sys.nelec-self.ncas_elec

        self.ncore = self.ncoreel
        self.nact = self.ncas_orbs
        self.nvirt = self.sys.nlrg-(self.ncore+self.nact) # This is different from nvirtual, which is in the single-reference sense (nvirt in the HF reference)
        self.nhole = self.ncore+self.nact
        self.npart = self.nact+self.nvirt
        
        self.core = slice(0,self.ncore)
        self.active = slice(self.ncore, self.ncore+self.ncas_orbs)
        self.virt = slice(self.ncore+self.ncas_orbs, self.sys.nlrg)
        self.hole = slice(0,self.ncore+self.ncas_orbs)
        self.part = slice(self.ncore, self.sys.nlrg)
        
        self.hc = self.core
        self.ha = self.active
        self.pa = slice(0,self.nact)
        self.pv = slice(self.nact,self.nact+self.nvirt)

        self.hh = self.hole
        self.pp = slice(0,self.npart)

        self.e_casci_frzc = 0.0

        if (self.ncore != 0):
            self.e_casci_frzc = np.einsum('ii->',self.H1body[self.core,self.core]) + 0.5*np.einsum('ijij->',self.H2body[self.core,self.core,self.core,self.core])
            self.H1body = self.H1body[self.active, self.active].copy() + np.einsum('ipjp->ij',self.H2body[self.active,self.core,self.active,self.core])
            self.H2body = self.H2body[self.active, self.active, self.active, self.active]

        self.nuclear_repulsion = self.sys.nuclear_repulsion + self.e_casci_frzc.real
        
    def diagonalize_p_space(self):
        self.p_hamil = form_cas_hamiltonian(self.H1body, self.H2body, self.p_space_det_strings, self.verbose, self.cas)
        self.p_space_eigvals, self.p_space_eigvecs = np.linalg.eigh(self.p_hamil)
        self.p_space_energy = self.p_space_eigvals[0].real + self.nuclear_repulsion
        self.rdm1 = get_1_rdm(self.p_space_det_strings, self.cas, self.p_space_eigvecs[:,0], self.verbose)
    
    def generate_fois(self):
        self.fois = set()
        for ipdet in self.p_space_det_strings:
            self.fois |= set(enumerate_determinants(ipdet, *self.cas, 1))
            self.fois |= set(enumerate_determinants(ipdet, *self.cas, 2))
        
        self.fois = np.array(list(self.fois - set(self.p_space_det_strings)))
    
    def get_energy_criteria(self):
        self.e_impt = np.zeros(len(self.fois))
        for i, fi in enumerate(self.fois):
            e_i = get_hamil_element(fi, fi, self.H1body, self.H2body, self.cas).real + self.nuclear_repulsion
            delta = e_i - self.p_space_energy
            vconj = get_H_IP(fi, self.p_space_det_strings, self.p_space_eigvecs[:,0], self.H1body, self.H2body, self.cas)
            self.e_impt[i] = delta/2 - np.sqrt(delta**2/4 + (vconj*np.conjugate(vconj)).real)
    
    def get_model_space(self):
        _e_f_argsort = np.argsort(np.abs(self.e_impt)) # numpy sorts in increasing order (smallest first), which is what we want here
        _esum = .0
        for i in range(len(self.fois)):
            _esum += np.abs(self.e_impt[_e_f_argsort[i]])
            if (_esum >= self.sigma): break 

        if (i < len(self.fois) and self.pt2): self.e_pt2 = np.sum(self.e_impt[_e_f_argsort[:i]]) # we don't want to discard the last determinant

        self.m_space_det_strings = np.array(list(set(self.p_space_det_strings) | set(self.fois[_e_f_argsort[i:]])))

    def diagonalize_model_space(self):
        self.m_hamil = form_cas_hamiltonian(self.H1body, self.H2body, self.m_space_det_strings, self.verbose, self.cas)
        self.m_space_eigvals, self.m_space_eigvecs = np.linalg.eigh(self.m_hamil)
        self.m_space_energy = self.m_space_eigvals[0].real + self.nuclear_repulsion

    def coarse_grain_m_space(self):
        _ci = np.real(np.conj(self.m_space_eigvecs[:,0]) * self.m_space_eigvecs[:,0])
        _ci_argsort = np.argsort(_ci)[::-1] # numpy sorts in increasing order, we want decreasing
        _cisum = .0
        for i in range(len(_ci)):
            _cisum += _ci[_ci_argsort[i]]
            if (_cisum >= (1 - self.gamma*self.sigma)): break

        self.p_space_det_strings = self.m_space_det_strings[_ci_argsort[:i+1]]
        self.ndets = i+1

    def generate_cationic_basis(self):
        self.cationic_basis = set()
        for idet in self.m_space_det_strings:
            for iorb in range(self.cas[1]):
                _ann_i = annop(idet, iorb)
                if (_ann_i[0] != 0):
                    self.cationic_basis.add(_ann_i[1])

        self.cationic_basis = np.sort(list(self.cationic_basis))

    def run_tdaci(self, dt, nsteps, orb, propagator='exact'):
        self.dt = dt/24.18884327 # attoseconds to a.u. https://en.wikipedia.org/wiki/Hartree_atomic_units
        self.generate_cationic_basis()
        self.cationic_hamil = form_cas_hamiltonian(self.H1body, self.H2body, self.cationic_basis, self.verbose, self.cas)
        self.psi_t = self.generate_initial_state(self.m_space_eigvecs[:,0], orb)
        self.prepare_hole_occnum()

        if (propagator == 'exact'):
            self.cationic_eigvals, self.cationic_eigvecs = np.linalg.eigh(self.cationic_hamil)
            # cf. Helgaker Eq. 3.1.27
            self.propagator = np.einsum('ij,jk,lk->il', self.cationic_eigvecs, np.diag(-1j*self.dt*self.cationic_eigvals), self.cationic_eigvecs.conj(), optimize="optimal")
        if (propagator == 'rk4'):
            self.propagator = np.diag(np.ones(self.cationic_hamil.shape[0])) - (1j)*self.dt*self.cationic_hamil
            coeff = np.array([0.5*self.dt**2, self.dt**3*(-1j)/6, self.dt**4/24], dtype='complex128')
            for idx, pow in enumerate([2,3,4]):
                self.propagator += coeff[idx] * np.linalg.matrix_power(self.cationic_hamil, pow)

        self.hole_occnum = np.zeros(self.cas[1])
        for istep in range(nsteps):
            self.psi_t = np.einsum('ij,j->i', self.propagator, self.psi_t, optimize="optimal")
            
            self.hole_occnum = self.get_hole_occnum()

    def prepare_hole_occnum(self):
        # Essentially the 1-rdm code but only calculate the diagonal, so we can pre-determine which basis states will contribute
        _i_occlist = {i: set() for i in range(self.cas[1])}
        for idet, det in enumerate(self.cationic_basis):
            for i in range(self.cas[1]):
                if (test_bit(det,i)==1):
                    _i_occlist[i].add(idet)
                
        self.i_occlist = {int(i): np.sort(list(_i_occlist[i])) for i in range(self.cas[1])} # convert from set to list

    def get_hole_occnum(self):
        self.hole_occnum = [np.dot((self.psi_t[self.i_occlist[i]]).conj(), self.psi_t[self.i_occlist[i]]).real for i in range(self.cas[1])]

    def generate_initial_state(self, psi, orb):
        _ann_psi, _ann_basis = annihilate_state(psi, orb, self.m_space_det_strings)
        _psi_t0 = np.zeros(len(self.cationic_basis), dtype='complex128')

        # Now we need to align this to the complete cationic basis
        icat = 0
        for iann in range(len(_ann_basis)):
            while (self.cationic_basis[icat] != _ann_basis[iann]):
                icat += 1

            _psi_t0[icat] = _ann_psi[iann]
            icat += 1

        return _psi_t0/(np.dot(_psi_t0.conj(), _psi_t0)) # normalize it

    def do_propagation(self, psi_t, propagator):
        return np.einsum('ij,j->i', propagator, psi_t, optimize="optimal")
    
    def run_aci(self):
        _e_old = 0.0
        _e_pt2_old = 0.0

        for iter in range(self.maxiter):
            _t0 = time.time()
            
            self.iter = iter + 1
            print(f'Iteration {self.iter}')

            print(f'number of p space determinants: {self.ndets}')
            self.diagonalize_p_space()
            print(f'p space energy: {self.p_space_energy}')

            self.generate_fois()
            print(f'number of fois determinants: {len(self.fois)}')

            self.get_energy_criteria()

            self.get_model_space()
            print(f'number of model space determinants: {len(self.m_space_det_strings)}')
            if (self.pt2): print(f'pt2 energy: {self.e_pt2}')
            
            self.diagonalize_model_space()
            print(f'model space energy: {self.m_space_energy}')
            self.e_aci = self.m_space_energy
            
            if (self.pt2): 
                self.e_aci_pt2 = self.m_space_energy + self.e_pt2
                print(f'pt2-corrected model space energy: {self.e_aci_pt2}')
            
            if (np.abs(self.e_aci - _e_old) < self.etol):
                if (self.pt2):
                    if (np.abs(self.e_aci_pt2 - _e_pt2_old) < self.etol):
                        break
                else:
                    break

            _e_old = self.e_aci
            if (self.pt2): _e_pt2_old = self.e_aci_pt2
            
            self.coarse_grain_m_space()

            _t1 = time.time()

            print(f'Time taken: {_t1-_t0:.5f}\n')
        
        self.aci_rdm1 = get_1_rdm(self.m_space_det_strings, self.cas, self.m_space_eigvecs[:,0], self.verbose)

def davidson_solver(hamil,nroots,maxdim, maxiter):
    L = nroots*2
    ndets = hamil.shape[0]
    b = np.zeros((ndets, maxdim), dtype='complex128')
    b[:L,:L] = np.eye(L)
    res = np.zeros(ndets, dtype='complex128')
    hamdiag = np.diag(hamil)
    c = np.zeros((ndets,nroots), dtype='complex128')

    for iter in range(maxiter):
        b[:,:L], r = np.linalg.qr(b[:,:L])
        G = np.einsum('ni,nm,mj->ij',b[:,:L].conj(),hamil,b[:,:L]) # conj comes from the definition of the inner product
        lamb, alfa = np.linalg.eigh(G) # This should really be eig, right?

        
        if (L+nroots <= maxdim):
            for k in range(nroots):
                res = np.einsum('i,mn,ni->m',alfa[:,k],hamil,b[:,:L],optimize='optimal') - lamb[k]*np.einsum('i,ni->n',alfa[:,k],b[:,:L],optimize='optimal')
                delta = res / (lamb[k]-hamdiag)
                delta /= np.linalg.norm(delta)
                b[:,L+k] = delta
            
            L += nroots
        else:
            print('Collapse!')
            # subspace collapse
            c = np.einsum('ik,mi->mk',alfa[:,:nroots],b[:,:L])

            for k in range(nroots):
                res = 0.0j
                for i in range(L):
                    res += alfa[i,k]*(np.dot(hamil,b[:,i]) - lamb[k]*b[:,i])
                delta = res / (lamb[k]-hamdiag)
                delta /= np.linalg.norm(delta)
                b[:,nroots+k] = delta
            b[:,:nroots] = c
            L = nroots*2

    return lamb[:nroots], alfa[:,:nroots]

if (__name__=='__main__'):
    mol = pyscf.gto.M(
        verbose = 2,
        atom = '''
    H 0 0 0
    F 0 1.5 0
    ''',
        basis = 'cc-pvdz', spin=0, charge=0, symmetry=False
    )
    a = RelForte(mol, verbose=True, density_fitting=False, decontract=False)
    a.run_dhf(transform=True, debug=True, frozen=(2,0), dump_mo_coeff=False, algo='direct', with_gaunt=True)