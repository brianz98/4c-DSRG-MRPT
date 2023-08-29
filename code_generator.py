import wicked as w
import numpy as np
import re
from fractions import Fraction
import time

w.reset_space()
w.add_space("c", "fermion", "occupied", list('ijklmn'))
w.add_space("a", "fermion", "general", list('uvwxyzrst'))
w.add_space("v", "fermion", "unoccupied", list('abcdef'))
wt = w.WickTheorem()

def split_single_tensor(tensor):
    """
    Split a single expression or string of the form
    >>> H^{a0,a3}_{a1,a2}
    into a tuple containing its label, the upper, and lower indices, i.e.,
    >>> ('H', [a0, a3], [a1, a2])
    If the label is 'F' or 'V', the upper and lower indices are swapped 
    (see 4c-dsrg-mrpt2.ipynb::dsrg_mrpt2_update to see why)
    """
    tensor = str(tensor)
    label, indices = str(tensor).split('^{')
    upper = indices.split('}_')[0].split(',')
    lower = indices.split('_{')[1].split('}')[0].split(',')
    if label in ['F', 'V']: 
        upper, lower = lower, upper
    return label, upper, lower

def get_unique_tensor_indices(tensor, unused_indices, index_dict):
    """
    >>> ('V', [a0, c0], [a1, a2]) -> 'uiwx'
    Get the indices of a tensor for use in an einsum contraction.
    For example, given the tensor from split_single_tensor, 
    the list of available indices and indices that have already been assigned, 
    we can generate the index string.
    """
    label, upper, lower = tensor
    indstr = ''
    for i in upper+lower:
        if index_dict.get(i):
            indstr += index_dict[i]
        else:
            index = unused_indices[i[0]].pop(0)
            index_dict[i] = index
            indstr += index
    
    return indstr

def get_tensor_slice(tensor, fmt):
    """
    >>> ('V', [a0, c0], [a1, a2]) -> "V['acaa']" (fmt = 'dict')
    >>> ('V', [a0, c0], [a1, a2]) -> "V[a,c,a,a]" (fmt = 'slice')
    """
    if tensor[0] in ['gamma1', 'eta1', 'lambda2', 'lambda3']:
        return tensor[0]
    else:
        if (fmt == 'dict'):
            return tensor[0] + "['" + ''.join([i[0] for i in tensor[1]]) + ''.join([i[0] for i in tensor[2]]) + "']"
        elif (fmt == 'slice'):
            if ('T' in tensor[0]): # T tensors are always particle-hole sized.
                return tensor[0] + '[' + ','.join(['h'+i[0] for i in tensor[1]]) + ',' + ','.join(['p'+i[0] for i in tensor[2]]) + ']'
            else:
                return tensor[0] + '[' + ','.join([i[0] for i in tensor[1]]) + ',' + ','.join([i[0] for i in tensor[2]]) + ']'

def get_lhs_tensor_name(tensor):
    """
    >>> ('V', [a0, c0], [a1, a2]) -> "Vacaa"
    """
    return tensor[0] + '_' + ''.join([i[0] for i in tensor[1]]) + ''.join([i[0] for i in tensor[2]])

def get_factor(expression):
    """
    Returns the prefactor of a right hand side expression, taking care of edge cases where an empty space (for +1.0) or a negative sign (for -1.0) are present.
    """
    factor = str(expression).split('+=')[-1].split(' ')[1]
    try:
        return float(Fraction(factor))
    except ValueError:
        if factor == '-':
            return -1.0
        else:
            return 1.0

def compile_einsum(expression, fmt='dict', tensor_name=None):
    """
    Compile an expression into a valid einsum expression.
    Turns a Wick&d expression (wicked._wicked.Equation) like H^{c0,a0}_{a1,a2} += 1/4 T2^{c0,a0}_{a3,a4} V^{a5,a6}_{a1,a2} eta1^{a4}_{a6} eta1^{a3}_{a5}
    into the einsum code string "H_caaa += +0.25000000 * np.einsum('iuvw,xyzr,wr,vz->iuxy', T2['caaa'], V['aaaa'], eta1, eta1, optimize='optimal')"
    """
    unused_indices = {'a':list('uvwxyzrst'), 'c':list('ijklmn'), 'v':list('abcdef')}
    index_dict = {}

    lhs = expression.lhs()
    rhs = expression.rhs()

    factor = get_factor(expression)

    exstr = ''  # holds the expression part of the einsum contraction, e.g., iuvw,xyzr,wr,vz->iuxy 
    tenstr = '' # holds the tensor label part of the einsum contraction, e.g., T2['caaa'], V['aaaa'], eta1, eta1, optimize='optimal')

    for i in str(rhs).split(' '):
        _ = split_single_tensor(i)
        tenstr += get_tensor_slice(_, fmt) + ', '
        exstr += get_unique_tensor_indices(_, unused_indices, index_dict)
        exstr += ','

    exstr = exstr[:-1] + '->'
    tenstr += "optimize='optimal')"

    _ = split_single_tensor(lhs)
    if (_[1] != ['']):
        left = get_lhs_tensor_name(_)
        res_indx = get_unique_tensor_indices(_, unused_indices, index_dict)
        exstr += res_indx
    else:
        left = _[0] # If it's scalar, just return the label.

    if tensor_name is not None:
        left = tensor_name

    einsumstr = left \
        + ' ' \
        + f"+= {factor:+.8f} * np.einsum('"\
        + exstr \
        + "', " \
        + tenstr

    return einsumstr    

def make_code(mbeq, fmt='dict', tensor_name=None):
    code = ''
    nlines = 0
    for i in mbeq:
        einsum = compile_einsum(i, fmt, tensor_name)
        code += '\t' + einsum + '\n'
        nlines += 1
    return code, nlines

def get_many_body_equations(op1, op2, nbody):
    """
    Returns the elements of the commutator of two operators.
    """
    comm = w.commutator(op1, op2)
    comm_expr = wt.contract(comm, nbody*2, nbody*2)
    return comm_expr.to_manybody_equation("H")

def make_nbody_elements(op1, op2, nbody, fmt='slice'):
    """
    Returns the elements of the commutator of two operators in einsum format.
    """
    code = ''
    nlines = 0
    mbeq = get_many_body_equations(op1, op2, nbody)

    for key in mbeq.keys():
        if (nbody != 0):
            if (fmt == 'slice'):
                lhs_slice = re.findall(r'[a-zA-Z]', key)
                if (len(lhs_slice) == 4):
                    lhs_slice = ','.join(lhs_slice[:2]) + ',' + ','.join(lhs_slice[2:][::-1]) # H^ij_ab -> H[i,j,b,a]
                else:
                    lhs_slice = ','.join(lhs_slice)
            _ = make_code(mbeq[key], fmt, f'C{nbody}[{lhs_slice}]')
        else:
            _ = make_code(mbeq[key], fmt, f'C{nbody}')
        code += _[0]
        nlines += _[1]
    return code, nlines

if __name__ == "__main__":
    t0 = time.time()
    F = w.utils.gen_op('F',1,'cav','cav',diagonal=True)
    V = w.utils.gen_op('V',2,'cav','cav',diagonal=True)
    Hop = F + V

    T1op = w.utils.gen_op('T1',1,'av','ca',diagonal=False)
    T2op = w.utils.gen_op('T2',2,'av','ca',diagonal=False)
    Top = T1op + T2op

    input_dict = {
        'H_T_C0':   (Hop, Top, 0),
        'H1_T1_C1': (F, T1op, 1),
        'H1_T2_C1': (F, T2op, 1),
        'H2_T1_C1': (V, T1op, 1),
        'H2_T2_C1': (V, T2op, 1),
        'H1_T2_C2': (F, T2op, 2),
        'H2_T1_C2': (V, T1op, 2),
        'H2_T2_C2': (V, T2op, 2)
    }
    slicedef = '\thc = mf.hc\n\tha = mf.ha\n\tpa = mf.pa\n\tpv = mf.pv\n\tc = mf.core\n\ta = mf.active\n\tv = mf.virt\n'

    with open('wicked_contractions.py', 'w') as f:
        f.write('import numpy as np\nimport time\n\n')
        for key in input_dict.keys():
            code, nlines = make_nbody_elements(*input_dict[key], fmt='slice')
            f.write('def ' + key + '(C1, C2, F, V, T1, T2, gamma1, eta1, lambda2, lambda3, mf, verbose=False):\n')
            f.write('\t# ' + str(nlines) + ' lines\n')
            f.write('\tt0 = time.time()\n')
            if (key == 'H_T_C0'): f.write('\tC0 = .0j\n')
            f.write(slicedef+'\n')
            f.write(code + '\n')
            f.write('\tt1 = time.time()\n')
            f.write('\tif verbose: print("'+key+' took {:.4f} seconds to run.".format(t1-t0))\n\n')
            if (key == 'H_T_C0'): f.write('\treturn C0\n\n')

    print("wicked_contractions.py generated in {:.5f} seconds.".format(time.time()-t0))