import ctypes
import subprocess
import tempfile
import os.path
from typing import Any
from .pyhephaistos import RawBuffer, ByteTensor, Program

class ArrayBuffer(RawBuffer):
    """
    Helper class for creating buffer holding an array of a given type.
    """
    
    def __init__(self, type: ctypes.Structure, size: int):
        """
        Creates a typed array buffer of the given type and size.
        """
        super().__init__(ctypes.sizeof(type) * size)
        self._arr = (type * size).from_address(super().address)
    
    def __len__(self):
        return self._arr.__len__()

    def __getitem__(self, key):
        return self._arr.__getitem__(key)

    def __setitem__(self, key, item):
        return self._arr.__setitem__(key, item)
    
    def __iter__(self):
        return iter(self._arr)

class ArrayTensor(ByteTensor):
    """
    Helper class for creating a tensor big enough to hold an array of given type
    and size.
    """

    def __init__(self, type: ctypes.Structure, size: int):
        """
        Create a typed array tensor of given type and size.
        """
        super().__init__(ctypes.sizeof(type) * size)

class StructureBuffer(RawBuffer):
    """
    Helper class for creating typed buffers.
    """

    def __init__(self, type: ctypes.Structure):
        """
        Creates a typed buffer with the given type.
        """
        super().__init__(ctypes.sizeof(type))
        # bypass __setattr__ we are overriding
        super().__setattr__("_pointer", ctypes.cast(super().address, ctypes.POINTER(type)))
    
    def __getattribute__(self, __name: str):
        #special types: size, size_bytes
        if __name in ["size", "size_bytes"]:
            return super().__getattribute__(__name)
        # bypass __getattribute__ to prevent infinite loop
        return super().__getattribute__("_pointer").contents.__getattribute__(__name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        # bypass our own __getattribute__
        super().__getattribute__("_pointer").contents.__setattr__(__name, __value)

class StructureTensor(ByteTensor):
    """
    Helper class for creating tensors matching the size of a given struct.
    """

    def __init__(self, type: ctypes.Structure):
        """
        Creates a tensor with the exact size to hold the given structure.
        """
        super().__init__(ctypes.sizeof(type))

def bindParams(program: Program, params) -> None:
    """
    Helper function to bind a list of params in a program.
    """
    for i, p in enumerate(params):
        p.bindParameter(program, i)
