Files obtained from the original [BERT](https://github.com/google-research/bert) (tensorflow) repository. 

Annotated with named shapes and compacted using tensor shorthand operators.
- all dimension variables declared [once](https://github.com/ofnote/tsalib/blob/d9350073f8a610e776fa808c7e7574d17ac4a02f/models/bert/modeling.py#L81)
- shorthand notation (tsn) used to annotate throughout
- warp used extensively
- found mismatches between documented and runtime shapes in the original code
