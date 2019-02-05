Files obtained from the original [BERT](https://github.com/google-research/bert) (tensorflow) repository. 

Annotated with named shapes and compacted using tensor shorthand operators.
- all dimension variables declared [once](https://github.com/ofnote/tsalib/blob/d9350073f8a610e776fa808c7e7574d17ac4a02f/models/bert/modeling.py#L81)
- shorthand notation ([TSN](notebooks/shorthand.md)) and **warp** used extensively

Benefits
- Several cryptic, shape wrangling functions (`reshape_from_matrix`, `reshape_to_matrix`, `transpose_for_scores`)turn into convenient, lucid one-liners 
- The *flow* of shapes becomes far more apparent in the code (courtesy both shape annotations and `warp` tsn arguments)
- Avoid copying around dimension sizes as arguments (`get_dim_vars`  by name at any location)
- Found inconsistencies between documented and runtime shapes and duplicate definitions in the original code.

Code can be simplified and cleaned up further.