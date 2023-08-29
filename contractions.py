import numpy as np
import time 

def antisymmetrize_2(T, indices):
    # antisymmetrize the residual
    T_anti = np.zeros(T.shape, dtype='complex128')
    T_anti += np.einsum("ijab->ijab",T)
    if indices[0] == indices[1]:
        T_anti -= np.einsum("ijab->jiab",T)
        if indices[2] == indices[3]:
            T_anti += np.einsum("ijab->jiba",T)
    if indices[2] == indices[3]:    
        T_anti -= np.einsum("ijab->ijba",T)        
    return T_anti

def dsrg_HT(F, V, T1, T2, gamma1, eta1, lambda2, lambda3, mf):
    # All three terms for MRPT2/3 are the same, but just involve different F/V
    hc = mf.hc
    ha = mf.ha
    pa = mf.pa
    pv = mf.pv

    # all quantities are stored ^{hh..}_{pp..}
    # h = {c,a}; p = {a, v}
    E = 0.0
    E += +1.000 * np.einsum("iu,iv,vu->",F[hc,pa],T1[hc,pa],eta1,optimize="optimal")
    E += -0.500 * np.einsum("iu,ixvw,vwux->",F[hc,pa],T2[hc,ha,pa,pa],lambda2,optimize="optimal")
    E += +1.000 * np.einsum("ia,ia->",F[hc,pv],T1[hc,pv],optimize="optimal")
    E += +1.000 * np.einsum("ua,va,uv->",F[ha,pv],T1[ha,pv],gamma1,optimize="optimal")
    E += -0.500 * np.einsum("ua,wxva,uvwx->",F[ha,pv],T2[ha,ha,pa,pv],lambda2,optimize="optimal")
    E += -0.500 * np.einsum("iu,ivwx,uvwx->",T1[hc,pa],V[hc,ha,pa,pa],lambda2,optimize="optimal")
    E += -0.500 * np.einsum("ua,vwxa,vwux->",T1[ha,pv],V[ha,ha,pa,pv],lambda2,optimize="optimal")
    E += +0.250 * np.einsum("ijuv,ijwx,vx,uw->",T2[hc,hc,pa,pa],V[hc,hc,pa,pa],eta1,eta1,optimize="optimal")
    E += +0.125 * np.einsum("ijuv,ijwx,uvwx->",T2[hc,hc,pa,pa],V[hc,hc,pa,pa],lambda2,optimize="optimal")
    E += +0.500 * np.einsum("iwuv,ixyz,vz,uy,xw->",T2[hc,ha,pa,pa],V[hc,ha,pa,pa],eta1,eta1,gamma1,optimize="optimal")
    E += +1.000 * np.einsum("iwuv,ixyz,vz,uxwy->",T2[hc,ha,pa,pa],V[hc,ha,pa,pa],eta1,lambda2,optimize="optimal")
    E += +0.250 * np.einsum("iwuv,ixyz,xw,uvyz->",T2[hc,ha,pa,pa],V[hc,ha,pa,pa],gamma1,lambda2,optimize="optimal")
    E += +0.250 * np.einsum("iwuv,ixyz,uvxwyz->",T2[hc,ha,pa,pa],V[hc,ha,pa,pa],lambda3,optimize="optimal")
    E += +0.500 * np.einsum("ijua,ijva,uv->",T2[hc,hc,pa,pv],V[hc,hc,pa,pv],eta1,optimize="optimal")
    E += +1.000 * np.einsum("ivua,iwxa,ux,wv->",T2[hc,ha,pa,pv],V[hc,ha,pa,pv],eta1,gamma1,optimize="optimal")
    E += +1.000 * np.einsum("ivua,iwxa,uwvx->",T2[hc,ha,pa,pv],V[hc,ha,pa,pv],lambda2,optimize="optimal")
    E += +0.500 * np.einsum("vwua,xyza,uz,yw,xv->",T2[ha,ha,pa,pv],V[ha,ha,pa,pv],eta1,gamma1,gamma1,optimize="optimal")
    E += +0.250 * np.einsum("vwua,xyza,uz,xyvw->",T2[ha,ha,pa,pv],V[ha,ha,pa,pv],eta1,lambda2,optimize="optimal")
    E += +1.000 * np.einsum("vwua,xyza,yw,uxvz->",T2[ha,ha,pa,pv],V[ha,ha,pa,pv],gamma1,lambda2,optimize="optimal")
    E += -0.250 * np.einsum("vwua,xyza,uxyvwz->",T2[ha,ha,pa,pv],V[ha,ha,pa,pv],lambda3,optimize="optimal")
    E += +0.250 * np.einsum("ijab,ijab->",T2[hc,hc,pv,pv],V[hc,hc,pv,pv],optimize="optimal")
    E += +0.500 * np.einsum("iuab,ivab,vu->",T2[hc,ha,pv,pv],V[hc,ha,pv,pv],gamma1,optimize="optimal")
    E += +0.250 * np.einsum("uvab,wxab,xv,wu->",T2[ha,ha,pv,pv],V[ha,ha,pv,pv],gamma1,gamma1,optimize="optimal")
    E += +0.125 * np.einsum("uvab,wxab,wxuv->",T2[ha,ha,pv,pv],V[ha,ha,pv,pv],lambda2,optimize="optimal")
    return E

def Hbar_active_twobody_wicked(mf, F, V, T1, T2, gamma1, eta1):
    hc = mf.hc
    ha = mf.ha
    pa = mf.pa
    pv = mf.pv

    # all quantities are stored ^{hh..}_{pp..}
    # h = {c,a}; p = {a, v}
    _V = np.zeros((mf.nact,mf.nact,mf.nact,mf.nact), dtype='complex128')
    # Term 6
    _V += -0.500 * np.einsum("ua,wxva->wxuv",F[ha, pv], T2[ha,ha, pa,pv],optimize="optimal")
    # Term 7
    _V += -0.500 * np.einsum("iu,ixvw->uxvw",F[hc, pa], T2[hc,ha, pa,pa],optimize="optimal")
    # Term 8
    _V += -0.500 * np.einsum("iu,ivwx->wxuv",T1[hc,pa], V[hc,ha, pa,pa],optimize="optimal")
    # Term 9
    _V += -0.500 * np.einsum("ua,vwxa->uxvw",T1[ha,pv], V[ha,ha, pa,pv],optimize="optimal")
    # Term 10
    _V += +0.125 * np.einsum("uvab,wxab->uvwx", T2[ha,ha, pv,pv], V[ha,ha, pv,pv],optimize="optimal")
    _V += +0.250 * np.einsum("uvya,wxza,yz->uvwx", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],eta1,optimize="optimal")
    # Term 11
    _V += +0.125 * np.einsum("ijuv,ijwx->wxuv", T2[hc,hc, pa,pa], V[hc,hc, pa,pa],optimize="optimal")
    _V += +0.250 * np.einsum("iyuv,izwx,zy->wxuv", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],gamma1,optimize="optimal")
    # Term 12
    _V += +1.000 * np.einsum("ivua,iwxa->vxuw", T2[hc,ha, pa,pv], V[hc,ha, pa,pv],optimize="optimal")
    _V += +1.000 * np.einsum("ivuy,iwxz,yz->vxuw", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],eta1,optimize="optimal")
    _V += +1.000 * np.einsum("vyua,wzxa,zy->vxuw", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],gamma1,optimize="optimal")

    return antisymmetrize_2(_V.conj(), 'aaaa')

def Hbar_active_onebody_wicked(mf, F, V, T1, T2, gamma1, eta1, lambda2):
    hc = mf.hc
    ha = mf.ha
    pa = mf.pa
    pv = mf.pv

    # all quantities are stored ^{hh..}_{pp..}
    # h = {c,a}; p = {a,v}
    _F = np.zeros((mf.nact,mf.nact), dtype='complex128')
    _F += -1.000 * np.einsum("iu,iv->uv",F[hc, pa],T1[hc,pa],optimize="optimal")
    _F += -1.000 * np.einsum("iw,ivux,xw->vu",F[hc, pa], T2[hc,ha, pa,pa],eta1,optimize="optimal")
    _F += -1.000 * np.einsum("ia,ivua->vu",F[hc, pv], T2[hc,ha, pa,pv],optimize="optimal")
    _F += +1.000 * np.einsum("ua,va->vu",F[ha, pv],T1[ha,pv],optimize="optimal")
    _F += +1.000 * np.einsum("wa,vxua,wx->vu",F[ha, pv], T2[ha,ha, pa,pv],gamma1,optimize="optimal")
    _F += -1.000 * np.einsum("iw,iuvx,wx->vu",T1[hc,pa], V[hc,ha, pa,pa],eta1,optimize="optimal")
    _F += -1.000 * np.einsum("ia,iuva->vu",T1[hc,pv], V[hc,ha, pa,pv],optimize="optimal")
    _F += +1.000 * np.einsum("wa,uxva,xw->vu",T1[ha,pv], V[ha,ha, pa,pv],gamma1,optimize="optimal")
    _F += -0.500 * np.einsum("ijuw,ijvx,wx->vu", T2[hc,hc, pa,pa], V[hc,hc, pa,pa],eta1,optimize="optimal")
    _F += +0.500 * np.einsum("ivuw,ixyz,wxyz->vu", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],lambda2,optimize="optimal")
    _F += -1.000 * np.einsum("ixuw,iyvz,wz,yx->vu", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],eta1,gamma1,optimize="optimal")
    _F += -1.000 * np.einsum("ixuw,iyvz,wyxz->vu", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],lambda2,optimize="optimal")
    _F += -0.500 * np.einsum("ijua,ijva->vu", T2[hc,hc, pa,pv], V[hc,hc, pa,pv],optimize="optimal")
    _F += -1.000 * np.einsum("iwua,ixva,xw->vu", T2[hc,ha, pa,pv], V[hc,ha, pa,pv],gamma1,optimize="optimal")
    _F += -0.500 * np.einsum("vwua,xyza,xywz->vu", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],lambda2,optimize="optimal")
    _F += -0.500 * np.einsum("wxua,yzva,zx,yw->vu", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],gamma1,gamma1,optimize="optimal")
    _F += -0.250 * np.einsum("wxua,yzva,yzwx->vu", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],lambda2,optimize="optimal")
    _F += +0.500 * np.einsum("iuwx,ivyz,xz,wy->uv", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],eta1,eta1,optimize="optimal")
    _F += +0.250 * np.einsum("iuwx,ivyz,wxyz->uv", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],lambda2,optimize="optimal")
    _F += -0.500 * np.einsum("iywx,iuvz,wxyz->vu", T2[hc,ha, pa,pa], V[hc,ha, pa,pa],lambda2,optimize="optimal")
    _F += +1.000 * np.einsum("iuwa,ivxa,wx->uv", T2[hc,ha, pa,pv], V[hc,ha, pa,pv],eta1,optimize="optimal")
    _F += +1.000 * np.einsum("uxwa,vyza,wz,yx->uv", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],eta1,gamma1,optimize="optimal")
    _F += +1.000 * np.einsum("uxwa,vyza,wyxz->uv", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],lambda2,optimize="optimal")
    _F += +0.500 * np.einsum("xywa,uzva,wzxy->vu", T2[ha,ha, pa,pv], V[ha,ha, pa,pv],lambda2,optimize="optimal")
    _F += +0.500 * np.einsum("iuab,ivab->uv", T2[hc,ha, pv,pv], V[hc,ha, pv,pv],optimize="optimal")
    _F += +0.500 * np.einsum("uwab,vxab,xw->uv", T2[ha,ha, pv,pv], V[ha,ha, pv,pv],gamma1,optimize="optimal")

    return _F.conj()