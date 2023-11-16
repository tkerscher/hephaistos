from __future__ import annotations
import numpy as np
from ctypes import Structure, c_float, c_uint32, c_int32
from typing import Type, Union
from hephaistos import Tensor


class vec(Structure):
    """Base class for GLSL vectors. Supports array indices and swizzling"""

    # def __new__(cls, *arg, **kwargs) -> Self:
    #     if hasattr(cls, "_fields_"):
    #         return super().__new__(cls, kwargs)
    #     if not 2 <= cls._dim_ <= 4:
    #         raise ValueError("Vector dimension must be 2,3 or 4!")
    #     cls._fields_ = [(d, cls._scalar_) for d in ["x", "y", "z", "w"][: cls._dim_]]
    #     return super().__new__(cls, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if len(args) == 1:
            self[:] = args[0]
        elif len(args) > 1:
            self[:] = args

    def __len__(self) -> int:
        return self._dim_

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(self._dim_)[idx])
        elif idx >= self._dim_:
            raise IndexError("vector has no such dimension")
        else:
            return getattr(self, ["x", "y", "z", "w"][idx])

    def __getattr__(self, name: str):
        if len(name) == 1:
            return super().__getattr__(name)
        else:
            return tuple(getattr(self, atr) for atr in name)

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            for i in range(self._dim_)[idx]:
                try:
                    self[i] = value[i]
                except:
                    self[i] = value
        elif idx >= self._dim_:
            raise IndexError("vector has no such dimension")
        else:
            return setattr(self, ["x", "y", "z", "w"][idx], value)

    def __setattr__(self, name: str, value) -> None:
        # special case: value property
        if name == "value":
            self[:] = value
        if len(name) == 1:
            return super().__setattr__(name, value)
        for i, atr in enumerate(name):
            try:
                super().__setattr__(atr, value[i])
            except:
                super().__setattr__(atr, value)

    @property
    def value(self):
        return tuple(self[i] for i in range(self._dim_))

    @value.setter
    def value(self, value):
        self[:] = value

    def __repr__(self) -> str:
        return str(tuple(self))


# Had some weird bug, where vec were constructed without any fields assigned
# this is a work around, but still have no idea why it's needed
def _initVecFields(scalar, dim):
    return [(["x","y","z","w"][d], scalar) for d in range(dim)]

class vec2(vec):
    _fields_ = _initVecFields(c_float, 2)
    _scalar_ = c_float
    _dim_ = 2


class vec3(vec):
    _fields_ = _initVecFields(c_float, 3)
    _scalar_ = c_float
    _dim_ = 3


class vec4(vec):
    _fields_ = _initVecFields(c_float, 4)
    _scalar_ = c_float
    _dim_ = 4


class ivec2(vec):
    _fields_ = _initVecFields(c_int32, 2)
    _scalar_ = c_int32
    _dim_ = 2


class ivec3(vec):
    _fields_ = _initVecFields(c_int32, 3)
    _scalar_ = c_int32
    _dim_ = 3


class ivec4(vec):
    _fields_ = _initVecFields(c_int32, 4)
    _scalar_ = c_int32
    _dim_ = 4


class uvec2(vec):
    _fields_ = _initVecFields(c_uint32, 2)
    _scalar_ = c_uint32
    _dim_ = 2


class uvec3(vec):
    _fields_ = _initVecFields(c_uint32, 3)
    _scalar_ = c_uint32
    _dim_ = 3


class uvec4(vec):
    _fields_ = _initVecFields(c_uint32, 4)
    _scalar_ = c_uint32
    _dim_ = 4


class mat(Structure):
    """Base class for GLSL matrices"""

    # def __new__(cls, *args) -> Self:
    #     if not hasattr(cls, "_fields_"):
    #         cls._fields_ = [("items", cls._type_ * cls._length_)]
    #     return super().__new__(cls)

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self[:] = args[0]
        elif len(args) > 1:
            self[:] = args

    def __len__(self) -> int:
        return self._length_

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(self._length_)[idx])
        return self.items[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, slice):
            for i in range(self._length_)[idx]:
                try:
                    self[i][:] = value[i]
                except:
                    self[i][:] = value
        else:
            self.items[idx][:] = value

    @property
    def value(self):
        return tuple(self[i] for i in range(self._length_))

    @value.setter
    def value(self, value):
        self[:] = value

    def __repr__(self) -> str:
        return "\n".join(str(self.items[i]) for i in range(self._length_))


class mat2(mat):
    _fields_ = [("items", vec2*2)]
    _type_ = vec2
    _length_ = 2


class mat3x2(mat):
    _fields_ = [("items", vec2*3)]
    _type_ = vec2
    _length_ = 3


class mat4x2(mat):
    _fields_ = [("items", vec2*4)]
    _type_ = vec2
    _length_ = 4


class mat2x3(mat):
    _fields_ = [("items", vec3*2)]
    _type_ = vec3
    _length_ = 3


class mat3(mat):
    _fields_ = [("items", vec3*3)]
    _type_ = vec3
    _length_ = 3


class mat4x3(mat):
    _fields_ = [("items", vec3*4)]
    _type_ = vec3
    _length_ = 4


class mat2x4(mat):
    _fields_ = [("items", vec4*2)]
    _type_ = vec4
    _length_ = 2


class mat3x4(mat):
    _fields_ = [("items", vec4*3)]
    _type_ = vec4
    _length_ = 3


class mat4(mat):
    _fields_ = [("items", vec4*4)]
    _type_ = vec4
    _length_ = 4


def stackVector(arrays, vecType: Type[vec]):
    """
    Stacks arrays to produce an array of type vecType, i.e. each array provides
    the data for the dimension corresponding to its index.

    Parameters
    ----------
    array: sequence of array_like
        Data for each dimension of the vectors

    vecType: Vector type
        Type of each column in the resulting array

    Returns
    -------
    stacked: ndarray, dtype=vecType
        The stacked array of given vector type

    Examples
    --------
    >>> x = np.array([1.0,2.0])
    >>> y = np.array([3.0,4.0])
    >>> z = np.array([5.0,6.0])
    >>> data = stackVector((x,y,z), vec3)
    >>> data
    array([(1., 3., 5.), (2., 4., 6.)],
      dtype={'names': ['x', 'y', 'z'],
      'formats': ['<f4', '<f4', '<f4'],
      'offsets': [0, 4, 8],
      'itemsize': 12, 'aligned': True})

    >>> buf = ArrayBuffer(vec3, 2)
    >>> buf.numpy()[:] = data
    >>> buf
    array([(1., 3., 5.), (2., 4., 6.)],
      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    """
    return (
        np.stack(arrays, axis=-1, dtype=np.dtype(vecType._scalar_))
        .ravel()
        .view(dtype=vecType)
    )


class buffer_reference(Structure):
    """packed buffer reference into a uvec2"""

    _fields_ = [("lo", c_uint32), ("hi", c_uint32)]

    def __init__(self, value=0):
        self.value = value

    @property
    def value(self) -> int:
        return self.lo + (self.hi << 32)

    @value.setter
    def value(self, value: Union[int, Tensor]) -> None:
        if isinstance(value, Tensor):
            value = value.address
        self.lo = value & 0xFFFFFFFF
        self.hi = value >> 32 & 0xFFFFFFFF
