from hephaistos.glsl import *
from hephaistos import ByteTensor


def test_vec2():
    v = vec2()
    assert tuple(v) == (0.0, 0.0)
    v = vec2(1.0)
    assert tuple(v) == (1.0, 1.0)
    v.y = -5.0
    assert v[:] == (1.0, -5.0)
    assert v.yy == (-5.0, -5.0)
    assert v.yx == (-5.0, 1.0)
    v[:] = -2.0
    assert v[:] == (-2.0, -2.0)
    v[1:] = -3.0
    assert v[:] == (-2.0, -3.0)
    v.value = 1.0
    assert v[:] == (1.0, 1.0)


def test_vec3():
    v = vec3()
    assert tuple(v) == (0.0, 0.0, 0.0)
    v = vec3(1.0)
    assert tuple(v) == (1.0, 1.0, 1.0)
    v.y = -5.0
    assert v[:] == (1.0, -5.0, 1.0)
    v.zx = (-2.0, 7.0)
    assert v.xxzy == (7.0, 7.0, -2.0, -5.0)
    v[1:] = 1.0
    assert v[:] == (7.0, 1.0, 1.0)
    v.value = 10.0
    assert v[:] == (10.0, 10.0, 10.0)


def test_vec4():
    v = vec4()
    assert tuple(v) == (0.0, 0.0, 0.0, 0.0)
    v = vec4(1.0)
    assert tuple(v) == (1.0, 1.0, 1.0, 1.0)
    v.z = -5.0
    assert v[:] == (1.0, 1.0, -5.0, 1.0)
    v.wx = (-2.0, 7.0)
    assert v.xwzy == (7.0, -2.0, -5.0, 1.0)
    v[1:] = 1.0
    assert v[:] == (7.0, 1.0, 1.0, 1.0)
    v.value = 10.0
    assert v[:] == (10.0, 10.0, 10.0, 10.0)


def test_ivec2():
    v = ivec2()
    assert tuple(v) == (0, 0)
    v = ivec2(1)
    assert tuple(v) == (1, 1)
    v.y = -5
    assert v[:] == (1, -5)
    assert v.yy == (-5, -5)
    assert v.yx == (-5, 1)
    v[:] = -2
    assert v[:] == (-2, -2)
    v[1:] = -3
    assert v[:] == (-2, -3)
    v.value = 1
    assert v[:] == (1, 1)


def test_ivec3():
    v = ivec3()
    assert tuple(v) == (0, 0, 0)
    v = ivec3(1)
    assert tuple(v) == (1, 1, 1)
    v.y = -5
    assert v[:] == (1, -5, 1)
    v.zx = (-2, 7)
    assert v.xxzy == (7, 7, -2, -5)
    v[1:] = 1
    assert v[:] == (7, 1, 1)
    v.value = 10
    assert v[:] == (10, 10, 10)


def test_ivec4():
    v = ivec4()
    assert tuple(v) == (0, 0, 0, 0)
    v = ivec4(1)
    assert tuple(v) == (1, 1, 1, 1)
    v.z = -5
    assert v[:] == (1, 1, -5, 1)
    v.wx = (-2, 7)
    assert v.xwzy == (7, -2, -5, 1)
    v[1:] = 1
    assert v[:] == (7, 1, 1, 1)
    v.value = 10
    assert v[:] == (10, 10, 10, 10)


def test_uvec2():
    v = ivec2()
    assert tuple(v) == (0, 0)
    v = ivec2(1)
    assert tuple(v) == (1, 1)
    v.y = 5
    assert v[:] == (1, 5)
    assert v.yy == (5, 5)
    assert v.yx == (5, 1)
    v[:] = 2
    assert v[:] == (2, 2)
    v[1:] = 3
    assert v[:] == (2, 3)
    v.value = 10
    assert v[:] == (10, 10)


def test_uvec3():
    v = uvec3()
    assert tuple(v) == (0, 0, 0)
    v = uvec3(1)
    assert tuple(v) == (1, 1, 1)
    v.y = 5
    assert v[:] == (1, 5, 1)
    v.zx = (2, 7)
    assert v.xxzy == (7, 7, 2, 5)
    v[1:] = 1
    assert v[:] == (7, 1, 1)
    v.value = 10
    assert v[:] == (10, 10, 10)


def test_uvec4():
    v = uvec4()
    assert tuple(v) == (0, 0, 0, 0)
    v = uvec4(1)
    assert tuple(v) == (1, 1, 1, 1)
    v.z = 5
    assert v[:] == (1, 1, 5, 1)
    v.wx = (2, 7)
    assert v.xwzy == (7, 2, 5, 1)
    v[1:] = 1
    assert v[:] == (7, 1, 1, 1)
    v.value = 10
    assert v[:] == (10, 10, 10, 10)


def test_mat3x2():
    m = mat2x4(2.0)
    assert len(m) == 2  # columns
    assert m[0][:] == (2.0, 2.0, 2.0, 2.0)
    assert m[1][:] == (2.0, 2.0, 2.0, 2.0)
    m[0][2] = 4.0
    assert m[0][:] == (2.0, 2.0, 4.0, 2.0)
    assert m[1][:] == (2.0, 2.0, 2.0, 2.0)
    m[1] = -1.0
    assert m[0][:] == (2.0, 2.0, 4.0, 2.0)
    assert m[1][:] == (-1.0, -1.0, -1.0, -1.0)
    m[0] = (1.0, 2.0, 3.0, 4.0)
    assert m[0][:] == (1.0, 2.0, 3.0, 4.0)
    assert m[1][:] == (-1.0, -1.0, -1.0, -1.0)
    m.value = 10.0
    assert m[0][:] == (10.0, 10.0, 10.0, 10.0)
    assert m[1][:] == (10.0, 10.0, 10.0, 10.0)


def test_buffer_reference():
    tensor = ByteTensor(64)
    ref = buffer_reference(tensor)
    assert ref.value == tensor.address
    t2 = ByteTensor(64)
    ref.value = t2
    assert ref.value == t2.address
    assert (t2.address & 0xFFFFFFFF) == ref.lo
    ref.value = 1564564
    assert ref.value == 1564564


def test_composite():
    class A(Structure):
        _fields_ = [("a", vec2), ("b", ivec3), ("m", mat3x4)]

    a = A()

    a.a[:] = 1.0
    a.b[:] = -5
    a.m = mat3x4(3.0)

    assert a.a[:] == (1.0, 1.0)
    assert a.b[:] == (-5, -5, -5)
    assert a.m[0][:] == (3.0, 3.0, 3.0, 3.0)
    assert a.m[1][:] == (3.0, 3.0, 3.0, 3.0)
    assert a.m[2][:] == (3.0, 3.0, 3.0, 3.0)

    a.b[1] = 10
    assert a.b[:] == (-5, 10, -5)

    a.m[1] = (-1.0, -2.0, -3.0, -4.0)
    assert a.m[0][:] == (3.0, 3.0, 3.0, 3.0)
    assert a.m[1][:] == (-1.0, -2.0, -3.0, -4.0)
    assert a.m[2][:] == (3.0, 3.0, 3.0, 3.0)
