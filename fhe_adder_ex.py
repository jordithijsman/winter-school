from random import randint
from typing import List
import time

##### ignore this :)

def bit_decompose(a: int, n:int):
    bits = [0] * n
    for i in range(n):
        bits[i] = (a >> i) % 2
    return bits

def bit_compose(a: List[int]):
    return int("".join(map(str, a)), 2)

#####


from openfhe import *

def full_adder(context, a, b, c_in):

    if isinstance(c_in, int):
        assert c_in == 0 or c_in == 1
        if c_in == 0:
            # TODO
            pass
        else:
            # TODO
            pass
    else:
        # TODO
        pass

    return s, c_out

def carry_ripple_adder(context, a, b):

    # note, a[0] corresponds to the *least* significant bit of a (same for b)

    assert isinstance(a, list)
    assert isinstance(b, list)
    assert len(a) == len(b)

    n = len(a)

    s_0, c_in = full_adder(context, a[0], b[0], 0)
    s = [s_0]
    for i in range(1, n):
        s_i, c_in = full_adder(context, a[i], b[i], c_in)
        s.append(s_i)

    return s[::-1], c_in

if __name__ == "__main__":

    print("#" * 16 + " Carry Ripple FHE Adder " + "#" * 16)

    sec_level = STD128 # STD128_LMKCDEY
    br_alg = GINX # LMKCDEY

    cc = BinFHEContext()
    cc.GenerateBinFHEContext(sec_level, br_alg)

    sk = cc.KeyGen()

    print("Generating FHE keys...")
    
    cc.BTKeyGen(sk)

    bit_count = 16

    a_int = randint(0, 2**bit_count)
    b_int = randint(0, 2**bit_count)

    a_bits = bit_decompose(a_int, bit_count)
    b_bits = bit_decompose(b_int, bit_count)

    # encrypt all bits
    a_enc = [cc.Encrypt(sk, a_i) for a_i in a_bits]
    b_enc = [cc.Encrypt(sk, b_i) for b_i in b_bits]

    print("Performing addition...")
    print()
    print()
    start_add = time.time()

    s_enc, carry_enc = carry_ripple_adder(cc, a_enc, b_enc)
    
    end_add = time.time()
    elapsed = end_add - start_add

    # decrypt bits
    s_bits = [cc.Decrypt(sk, s_enc_i) for s_enc_i in s_enc]

    # decrypt carry
    c_out = cc.Decrypt(sk, carry_enc)

    s_int = bit_compose(s_bits)
    
    s_exp = (a_int + b_int) % 2**bit_count
    c_exp = 1 if (a_int + b_int) >= 2**bit_count else 0
    gates_eval = (bit_count - 1) * 5 + 2

    print(f"Operand A={a_int} B={b_int}. Using {bit_count} precision")
    print(f"Expected result S={s_exp}, C={c_exp}, got S={s_int}, C={c_out}")
    print(f"Total: addition took {elapsed:.2f} seconds.")
    print(f"Per-Gate: evaluated {gates_eval} gates or {elapsed / gates_eval: .2f} seconds per gate")
