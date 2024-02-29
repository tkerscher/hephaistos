import numpy as np

from hephaistos.queue import *
from ctypes import *


class Header(Structure):
    _fields_ = [("u", c_uint32), ("f", c_float)]


class Item(Structure):
    _fields_ = [("a", c_uint32), ("b", c_float), ("v", c_float * 3)]


def test_queueSize():
    size_queue = sizeof(Item) * 100
    size_header = sizeof(Header)
    size_count = sizeof(QueueView.Counter)
    assert queueSize(Item, 100) == size_queue + size_count
    assert queueSize(Item, 100, skipCounter=True) == size_queue
    assert queueSize(Item, 100, header=Header) == size_queue + size_count + size_header
    assert (
        queueSize(Item, 100, header=Header, skipCounter=True)
        == size_queue + size_header
    )


def test_QueueBuffer():
    buf = QueueBuffer(Item, 100, skipCounter=True)
    assert buf.size_bytes == queueSize(Item, 100, skipCounter=True)
    buf = QueueBuffer(Item, 100, header=Header, skipCounter=True)
    assert buf.size_bytes == queueSize(Item, 100, header=Header, skipCounter=True)
    buf = QueueBuffer(Item, 100, header=Header)
    assert buf.size_bytes == queueSize(Item, 100, header=Header)

    assert buf.item == Item
    assert buf.capacity == 100

    view = buf.view
    assert view.fields == {name for name, _ in Item._fields_}

    view.header.u = 32
    assert view.header.u == 32
    view.header.f = 5.0
    assert view.header.f == 5.0

    view["b"][:] = np.arange(100)
    assert (view["b"] == np.arange(100)).all()
    view["v"][:] = np.arange(300).reshape((-1, 3))
    assert (view["v"] == np.arange(300).reshape((-1, 3))).all()

    view["v"][10:20, :2] = 5.0
    exp = np.arange(300).reshape((-1, 3))
    exp[10:20, :2] = 5.0
    assert (view["v"] == exp).all()


def test_QueueTensor():
    ten = QueueTensor(Item, 100, skipCounter=True)
    assert ten.size_bytes == queueSize(Item, 100, skipCounter=True)
    ten = QueueTensor(Item, 100, header=Header, skipCounter=True)
    assert ten.size_bytes == queueSize(Item, 100, header=Header, skipCounter=True)
    ten = QueueTensor(Item, 100, header=Header)
    assert ten.size_bytes == queueSize(Item, 100, header=Header)

    assert ten.item == Item
    assert ten.capacity == 100


def test_dumpQueue():
    buffer = QueueBuffer(Item, 100)
    queue = buffer.view

    queue["b"][:] = np.arange(100)
    queue["v"][:] = np.arange(300).reshape((-1, 3))
    queue["v"][10:20, :2] = 5.0
    queue["a"][:] = np.arange(100).astype(np.uint32)
    queue.count = 100

    dump = dumpQueue(queue)

    assert dump.keys() == queue.fields
    for field in queue.fields:
        assert (dump[field] == queue[field]).all()

    # test if independent copy
    queue["b"][:] += 1000.0
    assert (dump["b"] != queue["b"]).all()


def test_updateQueue():
    buffer = QueueBuffer(Item, 100)
    queue = buffer.view

    queue["b"][:] = np.arange(100)
    queue["v"][:] = np.arange(300).reshape((-1, 3))
    queue["v"][10:20, :2] = 5.0
    queue["a"][:] = np.arange(100).astype(np.uint32)
    queue.count = 100

    dump = dumpQueue(queue)
    del dump["a"]

    copyBuffer = QueueBuffer(Item, 100)
    copy = copyBuffer.view
    updateQueue(copy, dump)

    assert (copy["b"] == queue["b"]).all()
    assert (copy["v"] == queue["v"]).all()
    assert (copy["a"] != queue["a"]).any()
    assert copy.count == queue.count


def test_queueSerialization(tmp_path):
    file = tmp_path / "test.bin"

    buffer = QueueBuffer(Item, 100, header=Header)
    queue = buffer.view

    queue.header.u = 32
    queue.header.f = -10.0

    queue["b"][:] = np.arange(100)
    queue["v"][:] = np.arange(300).reshape((-1, 3))
    queue["v"][10:20, :2] = 5.0
    queue["a"][:] = np.arange(100).astype(np.uint32)
    queue.count = 100

    saveQueue(file, buffer)

    copyBuf = QueueBuffer(Item, 100, header=Header)
    loadQueue(file, copyBuf)
    copy = copyBuf.view

    assert (copy["b"] == queue["b"]).all()
    assert (copy["v"] == queue["v"]).all()
    assert (copy["a"] == queue["a"]).all()
    assert copy.count == queue.count
    assert copy.header.u == queue.header.u
    assert copy.header.f == queue.header.f
