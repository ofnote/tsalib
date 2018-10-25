from typing import List
from sympy import symbols


def arith_op (op, s1, s2):
    assert isinstance(s1, TS)
    s1 = s1.e

    if op == 'add':
        res = s1 + s2
    elif op == 'mul':
        res = s1 * s2
    elif op == 'div':
        res = s1 // s2  
    else:
        return NotImplemented

    return TS(res)


class TS:
    '''
    The Tensor Shape Expression Class
    '''
    def __init__(self, v):
        self.e = None
        if isinstance(v, str):
            self.e = symbols(v)
        else:
            self.e = v

    def __add__(self, n): return arith_op('add', self, n)

    def __mul__(self, n): return arith_op('mul', self, n)

    def __div__(self, n): return arith_op('div', self, n)

    def __eq__(self, d):
        assert isinstance(d, TS)
        return self.e == d.e    

    def __repr__(self):
        s = str(self.e)
        return s

def declare_base_shapes ():
    B = TS('Batch')
    D = TS('Dim')  #embedding dim
    V = TS('Vocab')
    Ci = TS('InChannels')
    Co = TS('OutChannels')
    Dh = TS('HiddenDim')  #hidden dim inside encoder/decoder
    Te = TS('EncoderTime')  #time along encoder
    Td = TS('DecoderTime')  #time along decoder

    return B, D, V, Dh, Te, Td, Ci, Co
