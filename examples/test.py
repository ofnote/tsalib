import sys
sys.path.append('../')

from tsalib.ts_lite import TSLite
from tsalib.ts import TS, declare_base_shapes

B, D, V, Dh, Te, Td, Ci, Co = declare_base_shapes()
H = TS('Height')
W = TS('Width')
C = TS('Channels')

def testlite():

    B = TSLite('Batch')
    T = TSLite('EncoderTime')  #time along encoder
    D = TSLite('Dim')  #embedding dim
    V = TSLite('Vocab')
    Dh = TSLite('HiddenDim')  #hidden dim inside encoder/decoder
    Td = TSLite('DecoderTime')  #time along decoder

    import torch
    a: (B, D) = torch.Tensor([[1., 2.], [3., 4.]])
    print(a.size())
    b: (2, B, D) = torch.stack([a, a])
    print(b.size())

    K = D * 2
    print((2, B, D))


def test():
    B = TS('Batch')
    T = TS('Time')  #time along encoder
    D = TS('Dim')  #embedding dim
    V = TS('Vocab')
    Dh = TS('HiddenDim')  #hidden dim inside encoder/decoder
    Td = TS('DecoderTime')  #time along decoder
    Dl = TS('LangEmbDim')  #embedding dim

    import torch
    a: (B, D) = torch.Tensor([[1., 2.], [3., 4.]])
    print(a.size())
    b: (2, B, D) = torch.stack([a, a])
    print(b.size())
    K = D * 2
    print((2, B, K))




if __name__ == '__main__':
    #testlite() 
    test()
   