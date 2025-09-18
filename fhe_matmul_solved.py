from openfhe import *
from enum import Enum
from math import log, ceil
from time import time

import argparse
import numpy as np
import numpy.typing as npt

##### Helper Functions #####

#################################################################################################################
#################################################################################################################
#################################################################################################################

all_indices = []
sk = None

class AXPYMethod(Enum):
    LARGE_COLS = 0,
    LARGE_ROWS = 1,
    SQUARE = 2,
    SECRET_SQUARE = 3,
    SECRET_HUGE = 4

def Meth2Str(x: AXPYMethod):
    
    x_idx = x.value
    if not isinstance(x_idx, int):
        x_idx = x_idx[0]

    return ["LARGE_COLS","LARGE_ROWS","SQUARE","SQUARE (Precomp.)","SINGLE_MULT"][x_idx]

def EvalRotSlow(context, ct, idx:int):
    """
    Given a crypto context, a ciphertext and an index, computes
    a new ciphertext encoding the same coefficient vector but rotated by \idx slots
    We assume that rotation keys for all power-of-two indices were computed previously
    """

    global all_indices
    all_indices.append(idx)

    # return ct

    if idx == 0:
        return ct

    current_pow2 = 1
    idx_abs = abs(idx)
    sign = idx // idx_abs
    while idx_abs != 0:
        bit = idx_abs % 2
        if bit != 0:
            try:
                ct = context.EvalRotate(ct, sign * current_pow2)
            except Exception as E:
                print("Failed for ", idx, sign * current_pow2)
                raise E
        current_pow2 *= 2
        idx_abs //= 2
    return ct

def EvalRotFast(context, ct, idx:int):
    """ 
    Given a crypto context, a ciphertext and an index, computes
    a new coiphertext encoding the same coefficient vector but rotated by \idx slots.
    We assume that a rotation key for \idx has been computed previously
    """
    return context.EvalRotate(ct, idx)


def EvalRotate(context, ct, idx, slow_rotation:bool = True):

    # we flip the sign of idx. openfhe interprets a negative index as a rotation to the right
    # but to be consistent with numpy.roll, we interpret it the other way around, i.e. positive index -> rotation to the right
    idx = -idx
    if slow_rotation:
        return EvalRotSlow(context, ct, idx)
    else:
        return EvalRotFast(context, ct, idx)

def Vec2Plaintext(context, vector):
    """
    Applies to IFFT + Rounding step to transform a vector
    of floats into a format compatible with the scheme
    """
    return context.MakeCKKSPackedPlaintext(vector)

def GetVectorSize(context):
    """
    Returns the maximum number of slots in any ciphertext
    for \context
    """
    return context.GetRingDimension() // 2

def Extract(ptx, length, precision = 8):
    ptx.SetLength(length)
    fmt = ptx.GetFormattedValues(precision)
    vec_slice = fmt[fmt.index("(")+1:fmt.index(")")]
    vec_str_list = vec_slice.split(",")
    vec_list = [float(v) for v in vec_str_list[:-1]]
    return np.array(vec_list)

def PrettyPrint(prefix, ptx, length, precision = 8):
    ptx.SetLength(length)
    fmt = ptx.GetFormattedValues(precision)
    print(prefix, fmt)


def AXPYSecretSquare(context, matrix:npt.ArrayLike, ct, slow_rotation:bool = True):
    """
    Given a SQUARE matrix and a ciphertext encoding a vector,
    outputs a ciphertext containing the product of matrix and the vector
    """
    n_rows, n_cols = matrix.shape
    assert n_rows == n_cols
    
    # Sometimes the maximal vector size exceeds the actual vector size
    # In those cases, we need to "pad" the vector in such a way that a rotation
    # actually behaves like a rotation, not like a shift
    vec_size = GetVectorSize(context)
    if vec_size > n_cols:
        # Rotate the vector by its own length
        ct_pad = EvalRotate(context, ct, n_cols, slow_rotation = slow_rotation)
        # Add. Now ct = Enc(vec || vec)
        ct = context.EvalAdd(ct_pad, ct)

    # duplicate matrix for cyclic diagonal
    M = np.concatenate([matrix, matrix], axis = 1)

    # set up accumulation ciphertext
    vec_pt = Vec2Plaintext(context, np.diag(M, 0))
    ct_result = context.EvalMult(ct, vec_pt)

    MM = context.GetCyclotomicOrder()
    digits = context.EvalFastRotationPrecompute(ct)

    for i in range(1, n_cols):
        # vec is the cyclic, i-th off-diagonal vector
        vec = np.diag(M, i)
        vec_pt = Vec2Plaintext(context, vec)
        # Rotate the input vector to align coefficient indices
        ct_rot_i = context.EvalFastRotation(ct, i,MM, digits)
        ct_i = context.EvalMult(ct_rot_i, vec_pt)
        ct_result = context.EvalAdd(ct_result, ct_i)

    return ct_result

def AXPYSecretHuge(context, matrix:npt.ArrayLike, ct, slow_rotation:bool = True):
    """
    Given a SQUARE matrix and a ciphertext encoding a vector,
    outputs a ciphertext containing the product of matrix and the vector
    """
    
    vec_size = GetVectorSize(context)
    n_rows, n_cols = matrix.shape
    assert n_rows * n_cols <= vec_size
    
    # start by making the ciphertext \n_row- redundant
    n_blocks = 1
    while n_blocks < n_rows:
        ct_rot = EvalRotate(context, ct, n_blocks * n_cols, slow_rotation=slow_rotation)
        ct = context.EvalAdd(ct, ct_rot)
        n_blocks *= 2

    # set up accumulation ciphertext
    matrix_flat = matrix.ravel('C')
    vec_pt = Vec2Plaintext(context, matrix_flat)
    ct_result = context.EvalMult(ct, vec_pt)

    current_block_size = n_cols
    while current_block_size > 1:
        rot_amount = current_block_size // 2
        ct_rot = EvalRotate(context, ct_result, -rot_amount, slow_rotation=slow_rotation)
        ct_result = context.EvalAdd(ct_result, ct_rot)
        current_block_size //= 2

    # mask
    mask = [0.0] * n_rows * n_cols
    mask[0::n_rows] = [1.0] * n_rows
    mask_ptx = Vec2Plaintext(context, mask)

    ct_result = context.EvalMult(ct_result, mask_ptx)

    # compress it back
    big_step = n_rows * n_cols // 2
    small_step = n_rows // 2
    while small_step > 0:
        ct_rot = EvalRotate(context, ct_result, -big_step + small_step,slow_rotation=slow_rotation)
        ct_result = context.EvalAdd(ct_result, ct_rot)
        big_step //= 2
        small_step //= 2

    return ct_result 
#################################################################################################################
#################################################################################################################
#################################################################################################################


def GenerateRotationIndices(method: AXPYMethod, batch_size:int,n_rows:int,n_cols:int, slow_rotation:bool = True):
    """
    Generates the required rotation indices for any given AXPY method, batch_size and matrix shape
    """
    if slow_rotation and not (method == AXPYMethod.SECRET_SQUARE):

        # By using the slow rotation, we only need to generate
        # keys for all idx = +- 2^i
        rotation_indices = [-1, 1]
        next_index = 2
        
        while next_index < batch_size:
            rotation_indices.extend([-next_index, next_index])
            next_index *= 2
        return rotation_indices

    if method == AXPYMethod.SQUARE or method == AXPYMethod.SECRET_SQUARE:
        assert n_rows == n_cols

        # For the square method, we need to perform rotation once for all -idx, s.t. idx < n
        # and possibly for n itself when batch_size != n
        return [i for i in range(n_rows)] + [-n_rows]

    elif method == AXPYMethod.LARGE_ROWS:

        # These indices are there to rotate the i-th coefficient in the masked vector
        # into the constant coefficient before replicating the values
        rotation_indices = [i for i in range(n_cols)]
        
        # The indices added here are used for the actual replication
        current_rot = 1
        while current_rot < n_rows:
            rotation_indices.append(-current_rot)
            current_rot *= 2
        return rotation_indices

    elif method == AXPYMethod.LARGE_COLS:

        # indices used to move the row-wise inner products into a specific slot
        rotation_indices = [-i for i in range(1, n_rows)]
        current_rot = 1

        # indices the compute the reduction for the inner product
        while current_rot < n_cols:
            rotation_indices.append(current_rot)
            current_rot *= 2
        return rotation_indices

    elif method == AXPYMethod.SECRET_HUGE:

        rotation_indices = []

        ii = 1
        while ii < n_cols:
            rotation_indices.append(ii)
            ii *= 2
        ii = 1
        while ii < n_rows:
            rotation_indices.append(-ii * n_cols)
            ii *= 2

        big_step = n_rows * n_cols // 2
        small_step = n_rows // 2
        while small_step > 0:
            idx = -big_step + small_step
            rotation_indices.append(-idx)
            big_step //=2
            small_step //=2

        return rotation_indices


    print("Unknown AXPY method !!!")
    return []


def AXPYSquare(context, matrix:npt.ArrayLike, ct, slow_rotation:bool = True):
    """
    Given a SQUARE matrix and a ciphertext encoding a vector,
    outputs a ciphertext containing the product of matrix and the vector
    """
    n_rows, n_cols = matrix.shape
    assert n_rows == n_cols
    
    # Sometimes the maximal vector size exceeds the actual vector size
    # In those cases, we need to "pad" the vector in such a way that a rotation
    # actually behaves like a rotation, not like a shift
    vec_size = GetVectorSize(context)
    if vec_size > n_cols:
        # Rotate the vector by its own length
        ct_pad = EvalRotate(context, ct, n_cols, slow_rotation = slow_rotation)
        # Add. Now ct = Enc(vec || vec)
        ct = context.EvalAdd(ct_pad, ct)

    # duplicate matrix for cyclic diagonal
    M = np.concatenate([matrix, matrix], axis = 1)

    # set up accumulation ciphertext
    vec_pt = Vec2Plaintext(context, np.diag(M, 0))
    ct_result = context.EvalMult(ct, vec_pt)

    for i in range(1, n_cols):
        # vec is the cyclic, i-th off-diagonal vector
        vec = np.diag(M, i)
        vec_pt = Vec2Plaintext(context, vec)
        # Rotate the input vector to align coefficient indices
        ct_rot_i = EvalRotate(context, ct, -i, slow_rotation=slow_rotation)
        ct_i = context.EvalMult(ct_rot_i, vec_pt)
        ct_result = context.EvalAdd(ct_result, ct_i)

    return ct_result



def InnerProduct(context, ct, vec: npt.ArrayLike, slow_rotation:bool = True):
    """
    Given a vector v_0 encoded in ct and a plaintext vector,
    outputs a ciphertext encoding v_1 s.t.
    (v_1)_0 = <v_0, vec> and
    (v_1)_i = 0, i neq 0
    """

    # Compute coefficient-wise product / hadamard product
    vec_pt = Vec2Plaintext(context, vec)
    tmp = context.EvalMult(ct, vec_pt)
    
    # Rotate and add the vector \log_2(total_size) times until the sum is in 0-coefficient
    total_size = len(vec)
    while total_size > 1:
        rot_amount = total_size // 2
        tmp_rot = EvalRotate(context, tmp, -rot_amount, slow_rotation=slow_rotation)
        tmp = context.EvalAdd(tmp, tmp_rot)
        total_size //= 2

    # Note that tmp_0 indeed contains the inner product.
    # However, the other coefficients contain partial sums we do not want
    # Therefore, we apply a mask to zero-out everythin
    # but the constant coefficient
    output_mask = [0.0] * total_size
    output_mask[0] = 1.0
    output_mask_ptx = Vec2Plaintext(context, output_mask)

    return context.EvalMult(tmp, output_mask_ptx)


def AXPYLargeCols(context, matrix:npt.ArrayLike, ct, slow_rotation:bool = True):
    
    n_rows, n_cols = matrix.shape

    row_outputs = []
    # First, we compute the inner products row-by-row
    for i in range(n_rows):
        ip_i = InnerProduct(context, ct, matrix[i, :], slow_rotation=slow_rotation)
        row_outputs.append(ip_i)

    ct_result = row_outputs[0]

    # As all individual inner products are in the constant coefficient
    # We move them to the desired index and add them together
    # This is correct as we've made sure that all other coefficients are 0
    for i in range(1, n_rows):
        ct_i_rot = EvalRotate(context, row_outputs[i], i, slow_rotation=slow_rotation)
        ct_result = context.EvalAdd(ct_result, ct_i_rot)

    return ct_result

def ReplicateConstantCoef(context, ct, n:int, slow_rotation:bool = True):
    """
    Given a ciphertext encoding a vector v s.t v_0 = V, v_i = 0 else
    outputs a ciphertext encoding w s.t. w_[0:n-1] = V 
    """
    vec_size = GetVectorSize(context)
    assert vec_size >= n
    
    # Round to next power of 2
    n_next_pow2 = 2**int(ceil(log(n , 2)))
    
    ct_result = ct
    current_reps = 1

    # The replication works as follows:
    # At iteration i, we have that w^i_[0:2**i - 1] = V
    # So we can compute w^(i+1) = w^i + Rot(w^i, 2**i) => w^(i+1)_[0:2**(i+1) - 1] = V
    while current_reps < n_next_pow2:
        ct_rot = EvalRotate(context, ct_result, current_reps,slow_rotation = slow_rotation)
        ct_result = context.EvalAdd(ct_rot, ct_result)
        current_reps *= 2

    return ct_result

def AXPYLargeRows(context, matrix: npt.ArrayLike, ct, slow_rotation:bool = True):

    n_rows, n_cols = matrix.shape

    # In this loop we construct the replicated vectors
    col_cts = []
    for i in range(n_cols):

        # First, we define the mask needed to isolate the i-th coefficient ...
        mask = [0.0] * n_cols
        mask[i] = 1.0
        mask_ptx = Vec2Plaintext(context, mask)
        # ... and apply it
        ct_i = context.EvalMult(ct, mask_ptx)

        # Next, we move the i-th coefficient into the constant coefficient
        ct_i_rot = EvalRotate(context, ct_i, -i, slow_rotation=slow_rotation)

        # Finally, we replicate the constant coefficient n_rows times
        ct_i_rep = ReplicateConstantCoef(context, ct_i_rot, n_rows, slow_rotation=slow_rotation)
        col_cts.append(ct_i_rep)

    # We multiply the first replicated vector with the first column to use as accumulator
    col_0 = matrix[:, 0]
    col_0_ptx = Vec2Plaintext(context, col_0)
    ct_result = context.EvalMult(col_cts[0], col_0_ptx)

    # Multiply each replicated vector with the corresponding column and add to ct_result
    for i in range(1, n_cols):
        col_i = matrix[:, i]
        col_i_ptx = Vec2Plaintext(context, col_i)
        ct_prod_i = context.EvalMult(col_i_ptx, col_cts[i])
        ct_result = context.EvalAdd(ct_result, ct_prod_i)
    
    return ct_result


def AXPY(context, matrix:npt.ArrayLike, ct, method: AXPYMethod, slow_rotation: bool = True):
        
    if method == AXPYMethod.SQUARE:
        return AXPYSquare(context, matrix, ct, slow_rotation=slow_rotation)
    elif method == AXPYMethod.LARGE_COLS:
        return AXPYLargeCols(context, matrix, ct, slow_rotation=slow_rotation)
    elif method == AXPYMethod.LARGE_ROWS:
        return AXPYLargeRows(context, matrix, ct, slow_rotation=slow_rotation)
    elif method == AXPYMethod.SECRET_SQUARE:
        return AXPYSecretSquare(context, matrix, ct, slow_rotation=slow_rotation)
    elif method == AXPYMethod.SECRET_HUGE:
        return AXPYSecretHuge(context, matrix, ct, slow_rotation=slow_rotation)


    print("Unknown AXPY method !!!")
    return ct


def SetupContextForAXPY(method:AXPYMethod, mat_dim,  max_vec_size:int, slow_rotation: bool = True, security_level = HEStd_NotSet, multiplicative_depth = 10):

    log2size = int(log(max_vec_size, 2))
    assert 2**log2size == max_vec_size, "maximum vector size is not a power of 2 !!!"

    scale_mod_size = 50

    parameters = CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(multiplicative_depth)
    parameters.SetScalingModSize(scale_mod_size)
    parameters.SetSecurityLevel(security_level)
    parameters.SetRingDim(max_vec_size * 2)
    parameters.SetBatchSize(max_vec_size)

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    rotation_indices = GenerateRotationIndices(method, max_vec_size, mat_dim[0], mat_dim[1], slow_rotation=slow_rotation)

    keys = cc.KeyGen()

    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, rotation_indices)

    return cc, keys


def Check(method: AXPYMethod, n_rows:int, n_cols:int, target_err = 1e-8, slow_rotation: bool = True):
    
    meth_str = Meth2Str(method)

    # Set up matrix
    element_range = (0, 16)
    matrix = np.random.randint(*element_range, size=(n_rows, n_cols)).astype(np.float64)
    vec = np.random.randint(*element_range, size=n_cols).astype(np.float64)

    # maximum vector size
    max_vec_size = 2**14
    assert max_vec_size >= n_rows * n_cols

    print("[!!!] FHE Based Matrix Vector Multiplication")
    print("[!!] Parameters:")
    print(f"[!] Method = {meth_str}")
    print(f"[!] Matrix dimension m x n, m = {n_rows}, n = {n_cols}")
    print(f"[!] Using general / slow rotation ? {slow_rotation}")

    context, keys = SetupContextForAXPY(method, (n_rows, n_cols), max_vec_size, slow_rotation=slow_rotation)

    vec_pt = Vec2Plaintext(context, vec)
    ct = context.Encrypt(keys.publicKey, vec_pt)

    times = []
    for i in range(100):
        tic = time()

        ct_res = AXPY(context, matrix, ct, method, slow_rotation=slow_rotation)

        toc = time()

        times.append(toc-tic)
        print(toc-tic)

    print(times)
    # Check output
    res_pt = context.Decrypt(ct_res, keys.secretKey)
    
    result_vec = Extract(res_pt, n_rows)
    
    target_vec = matrix @ vec

    rel_err = np.linalg.norm(result_vec - target_vec) / np.linalg.norm(target_vec)

    if rel_err < target_err:
        print(f"Result matches expected output. Relative error is {rel_err:.2f}")
        print(f"Product took {toc-tic:.3f}s.")
    else:
        print(f"[!] Result appears incorrect !!! Relative error is {rel_err:.2f}")
        print("[!] Expected:")
        print(target_vec)
        print("[!] Got:")
        print(result_vec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstration of FHE based matrix-vector products")
    
    parser.add_argument('-m','--method',help="Which AXPY method to use",required=True, choices=["square","large_rows","large_cols","secret_square","secret_huge"])
    parser.add_argument('-nr','--num-rows',help="Number of rows for the matrix",type=int,required=True)
    parser.add_argument('-nc','--num-cols',help="Number of columns for the matrix",type=int, required=True)
    parser.add_argument('-fast','--fast-rotation',help="Whether to use the slower or fast rotation",action='store_true')
    parser.add_argument('-prec','--precision',help="Precision for relative error calculation",type=float, default=1e-8)

    args = parser.parse_args()

    num_rows = args.num_rows
    num_cols = args.num_cols

    method = AXPYMethod[args.method.upper()]
    slow_rotation = not args.fast_rotation
    precision = args.precision

    Check(method, num_rows, num_cols, target_err=precision, slow_rotation=slow_rotation)

