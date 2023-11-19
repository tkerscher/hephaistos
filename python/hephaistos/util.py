import ctypes
from typing import Any, Union
from numpy.ctypeslib import as_array
from .pyhephaistos import ByteTensor, RawBuffer


def printSize(size_bytes: int) -> str:
    """
    Creates a printable representation of the given size in bytes by using a
    sensible unit like KB or MB.
    """
    if size_bytes == 0:
        return "0B"
    names = ["B", "KB", "MB", "GB", "TB", "PB"]
    dim = (size_bytes.bit_length() - 1) // 10
    dim = min(dim, len(names) - 1)
    unit = 2 << (dim * 10) - 1
    return f"{size_bytes/unit:.1f} {names[dim]}"


def collectFields(struct: ctypes.Structure) -> list[tuple[str, ctypes._SimpleCData]]:
    """
    Returns a flat list of fields and their types for a given ctype structure.
    """
    fields = []
    for name, T in struct._fields_:
        if type(T) == type(ctypes.Structure):
            fields.extend(
                [(f"{name}.{subname}", subT) for subname, subT in collectFields(T)]
            )
        elif type(T) == type(ctypes.Array):
            # array of structs or simple types?
            if type(T._type_) == type(ctypes.Structure):
                fields.extend(
                    [
                        (f"{name}[{i}].{subname}", subT)
                        for i in range(T._length_)
                        for subname, subT in collectFields(T._type_)
                    ]
                )
            else:
                fields.extend([(f"{name}[{i}]", T._type_) for i in range(T._length_)])
        else:
            fields.append((name, T))
    return fields


def createFlatType(struct: ctypes.Structure) -> ctypes.Structure:
    """
    Creates a equivalent ctype structure from the given one where all nested
    types are flattened.
    """
    fields = collectFields(struct)

    class Flat(ctypes.Structure):
        _fields_ = fields

    return Flat


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
        self._flat = (createFlatType(type) * size).from_address(super().address)

    @property
    def array(self):
        return self._arr

    @property
    def flatarray(self):
        return self._flat

    def numpy(self, flat: bool = False):
        """
        Returns a structured numpy array sharing the same memory as the buffer.
        If flat is True, the field names correspond the the flattened ones, i.e.
        ["dir.x"] instead of ["dir"]["x"].
        """
        return as_array(self.flatarray if flat else self.array)

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

    def __init__(self, type: ctypes.Structure, size: int, mapped: bool = False):
        """
        Create a typed array tensor of given type and size.
        """
        super().__init__(ctypes.sizeof(type) * size, mapped)


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
        super().__setattr__(
            "_pointer", ctypes.cast(super().address, ctypes.POINTER(type))
        )

    def __getattribute__(self, __name: str):
        # special types: size, size_bytes
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

    def __init__(self, typeOrData: Union[ctypes.Structure, Any], mapped: bool = False):
        """
        Creates a tensor with the exact size to hold the given structure.
        """
        if type(typeOrData) == type(ctypes.Structure):
            super().__init__(ctypes.sizeof(typeOrData), mapped)
        else:
            super().__init__(
                ctypes.addressof(typeOrData), ctypes.sizeof(typeOrData), mapped
            )
