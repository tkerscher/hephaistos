from .pyhephaistos import *
from .util import (
    ArrayBuffer,
    ArrayTensor,
    StructureBuffer,
    StructureTensor,
)

# make the Program.bindParams more pythonic
def _bindParams(program: Program, *params, **namedparams) -> None:
    """
    Helper function to bind a list of params in a program.
    """
    for i, p in enumerate(params):
        p.bindParameter(program, i)
    
    names = [b.name for b in program.bindings]
    for name, p in namedparams.items():
        if name not in names:
            continue
        p.bindParameter(program, name)
#register function in class
Program.bindParams = _bindParams
