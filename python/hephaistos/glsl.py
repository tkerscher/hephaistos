import numpy as np
from ctypes import Structure, c_float, c_uint32, c_int32

def defineVector(scalar, N, name):
    """
    Helper function for defining new vector types

    Parameters
    ----------
    scalar: ctypes._CSimpleType
        scalar type of the vector, i.e. float for vec3
    N: [2..4]
        number of channels
    name: str
        name of vector type
    
    Returns
    -------
    vector: class
        the new vector type
    """
    names = ["x", "y", "z", "w"]
    class Vector(Structure):
        _fields_ = [(n,scalar) for n in names[:N]]
    Vector.__name__ = name
    Vector._scalar_ = scalar
    return Vector


vec2 = defineVector(c_float, 2, "vec2")
vec3 = defineVector(c_float, 3, "vec3")
vec4 = defineVector(c_float, 4, "vec4")

ivec2 = defineVector(c_int32, 2, "ivec2")
ivec3 = defineVector(c_int32, 3, "ivec3")
ivec4 = defineVector(c_int32, 4, "ivec4")

uvec2 = defineVector(c_uint32, 2, "uvec2")
uvec3 = defineVector(c_uint32, 3, "uvec3")
uvec4 = defineVector(c_uint32, 4, "uvec4")


mat2 = vec2 * 2
mat3x2 = vec2 * 3
mat4x2 = vec2 * 4

mat2x3 = vec3 * 2
mat3 = vec3 * 3
mat4x3 = vec3 * 4

mat2x4 = vec4 * 2
mat3x4 = vec4 * 3
mat4 = vec4 * 4


def stackVector(arrays, vecType):
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
    return np.stack(arrays, axis=-1, dtype=np.dtype(vecType._scalar_)).ravel().view(dtype=vecType)

def addVector(v1, v2, vecType):
    """
    Adds two vectors of given type and returns their element-wise sum.
    """
    return stackVector([v1[f[0]] + v2[f[0]] for f in vecType._fields_], vecType)

def scaleVector(v, s, vecType):
    """
    Scales a given vector and returns the element-wise multiplication.
    """
    return stackVector([v[f[0]] * s for f in vecType._fields_], vecType)
