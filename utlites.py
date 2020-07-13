import numpy as np

def biterr(inputs_bits,ori_bits):

    symbols = inputs_bits.shape[0]
    num_bits = inputs_bits.shape[1]
    total_bits = symbols*num_bits
    a=np.reshape(inputs_bits, (1, total_bits))
    b=np.reshape(ori_bits, (1, total_bits))
    errors = (a != b).sum()
    # errors = errors.astype(int)
    ber = errors/total_bits
    return ber