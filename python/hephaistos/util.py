import ctypes
from typing import Any
from .pyhephaistos import RawBuffer


class StructureBuffer(RawBuffer):
    """
    Helper class for creating types buffers.
    """

    def __init__(self, type: ctypes.Structure):
        """
        Creates a typed buffer with the given type.
        """
        super().__init__(ctypes.sizeof(type))
        # bypass __setattr__ we are overriding
        super().__setattr__("_pointer", ctypes.cast(super().address, ctypes.POINTER(type)))
    
    def __getattribute__(self, __name: str):
        # bypass __getattribute__ to prevent infinite loop
        return super().__getattribute__("_pointer").contents.__getattribute__(__name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        # bypass our own __getattribute__
        super().__getattribute__("_pointer").contents.__setattr__(__name, __value)
