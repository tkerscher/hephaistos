from __future__ import annotations

import numpy as np
import warnings
from contextlib import ExitStack

from ctypes import Structure, c_uint32, pointer, sizeof
from hephaistos import Buffer, ByteTensor, Command, RawBuffer, Tensor, clearTensor
from hephaistos.util import printSize
from numpy import ndarray
from numpy.ctypeslib import as_array

from numpy.typing import NDArray
from os import PathLike
from typing import Any, BinaryIO, Dict, Optional, Set, Type, Union


class QueueView:
    """
    View allowing structured access to a queue stored in memory.
    A queue starts with an header describing the amount of data it holds, as
    well as a structure for indirect dispatching a processing shader.

    The data itself is stored as a structure of array to allow coalesced memory
    access for increased performance. This structure is built using a structure
    describing a single item, i.e. the following conversion happens:

    `struct { A a; B b; ... ; Z z; }` -> `struct { A a[N]; B b[N]; ...; Z z[N]; }`

    where `N` is the specified capacity.

    Parameters
    ----------
    data: int
        address of the memory the view should point at
    item: Structure
        Structure describing a single item. Used internally to create a
        structure of arrays.
    capacity: int
        Maximum number of items the queue can hold
    skipCounter: bool, default=False
        Whether the queue should omit its counter
    header: Structure | None, default=None
        Optional header prefixing the queue
    
    Note
    ----
    Manipulating data in the queue does not update the counter, which is zero
    right after initialization.
    """

    Counter = c_uint32
    """Type of the queue counter"""

    def __init__(
        self,
        data: int,
        item: Type[Structure],
        capacity: int,
        *,
        skipCounter: bool = False,
        header: Optional[Type[Structure]] = None,
    ) -> None:
        # store item type
        self._item = item
        self._capacity = capacity
        self._counter = None
        self._header = None
        # check if we have header
        if header is not None:
            self._header = header.from_address(data)
            data += sizeof(header)
        # check if we need counter
        if not skipCounter:
            self._counter = self.Counter.from_address(data)
            data += sizeof(self.Counter)

        # create SoA and use it to create data access
        class SoA(Structure):
            _fields_ = [
                (
                    (name, t._type_ * capacity * t._length_)  # handle arrays
                    if hasattr(t, "_length_")  # check if array
                    else (name, t * capacity)
                )  # handle scalar
                for name, t in item._fields_  # iterate over all fields
            ]

        self._data = SoA.from_address(data)
        # create arrays for each field
        self._fields = {
            # use transpose to align the first index on both scalar and arrays
            # i.e. each array element is treated as its own field from the
            # perspective of the memory
            name: as_array(getattr(self._data, name)).T
            for name, _ in item._fields_
        }
        # create set of field names (don't want to expose dict_keys)
        self._field_names = set(self._fields.keys())

    def __len__(self) -> int:
        return self._capacity

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __getitem__(self, key: Any) -> Union[QueueSubView, NDArray]:
        if isinstance(key, (int, ndarray, slice)):
            return QueueSubView(self, key)
        elif not isinstance(key, str):
            raise KeyError("Unsupported key type")
        if key not in self:
            raise KeyError(f"No field with name {key}")
        return self._fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise KeyError("Unsupported key type")
        if key not in self:
            raise KeyError(f"No field with name {key}")
        self._fields[key][:] = value

    @property
    def capacity(self) -> int:
        """Capacity of the queue"""
        return self._capacity

    @property
    def fields(self) -> Set[str]:
        """Set of field names"""
        return self._field_names

    @property
    def header(self) -> Optional[Structure]:
        """Optional header. None if no header is present"""
        return self._header

    @property
    def hasCounter(self) -> bool:
        """True, if queue has a counter"""
        return self._counter is not None

    @property
    def count(self) -> int:
        """Number of items in queue. Equal to capacity if counter was skipped"""
        if self._counter is not None:
            return self._counter.value
        else:
            return self.capacity

    @count.setter
    def count(self, value: int) -> None:
        if self._counter is not None:
            self._counter.value = value
        else:
            raise ValueError("The queue has no counter!")

    @property
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self._item

    def __repr__(self) -> str:
        return f"QueueView: {self.item.__name__}[{self.capacity}]"


class QueueSubView:
    """Utility class for slicing and masking a QueueView"""

    def __init__(
        self,
        orig: Union[QueueView, QueueSubView],
        mask: Union[int, slice, NDArray],
    ) -> None:
        self._orig = orig
        self._mask = mask
        # query new length by applying mask to a random field
        self._count = (
            1 if isinstance(mask, int) else len(orig[next(iter(orig.fields))][mask])
        )

    def __len__(self) -> int:
        return self._count

    def __contains__(self, key: str) -> bool:
        return key in self._orig

    def __getitem__(self, key) -> Union[QueueView, NDArray]:
        if isinstance(key, (int, ndarray, slice)):
            return QueueSubView(self, key)
        elif not isinstance(key, str):
            raise KeyError("Unsupported key type")
        if key not in self:
            raise KeyError(f"No field with name {key}")
        return self._orig[key][self._mask]

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise KeyError("Unsupported key type")
        if key not in self:
            raise KeyError(f"No field with name {key}")
        self._orig[key][self._mask] = value

    @property
    def fields(self) -> Set[str]:
        """List of field names"""
        return self._orig.fields

    @property
    def header(self) -> Optional[Structure]:
        """Optional header. None if no header is present"""
        return self._orig.header

    @property
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self._orig.item

    def __repr__(self) -> str:
        return f"QueueSubView: {self.item.__name__}[{self._count}]"


def dumpQueue(queue: Union[QueueView, QueueSubView]) -> Dict[str, NDArray]:
    """
    Creates a dictionary containing an entry for each field of the queue mapped
    to a copy of the corresponding data.
    """
    return {field: queue[field][: queue.count].copy() for field in queue.fields}


def updateQueue(
    queue: Union[QueueView, QueueSubView],
    data: Dict[str, NDArray],
    *,
    updateCount: bool = True,
) -> None:
    """
    Updates the given queue using data from the provided dictionary by matching
    keys with field names.

    Parameters
    ----------
    queue: QueueView | QueueSubView
        queue to update
    data: { field: NDArray }
        data used to update the queue
    updateCount: bool = True
        Whether to update the queue's count. Ignored if queue has no counter.

    Note
    ----
    If data contains arrays of varying length, warnings are produced and the
    counter will be updated to the smallest length capped to the queue's
    capacity.
    """
    counts = []
    for field, arr in data.items():
        # safety check
        if field not in queue.fields:
            warnings.warn(f'Skipping unknown field "{field}"')
            continue  # skip unknown field
        if len(arr) > queue.capacity:
            warnings.warn(f'Field "{field}" truncated to queue\'s capacity')
        # store data
        n = min(queue.capacity, len(arr))
        queue[field][:n] = arr[:n]
        counts.append(n)
    # check all fields had the same size
    if len(counts) > 0 and not all(c == counts[0] for c in counts):
        warnings.warn("Not all fields have the same length!")
    # update counter if necessary
    if queue.hasCounter and updateCount:
        queue.count = min(counts)


def queueSize(
    item: Union[Type[Structure], int],
    capacity: int,
    *,
    header: Optional[Type[Structure]] = None,
    skipCounter: bool = False,
) -> int:
    """
    Calculates the required size to store a queue of given size and item type

    Parameters
    ----------
    item: Structure | int
        Either a Structure describing a single item or the size of it
    capacity: int
        Maximum number of items the queue can hold
    header: Structure | None, default = None
        Optional header prefixing the queue
    skipCounter: bool, default=False
        True, if the queue should not contain a counter
    """
    itemSize = item if isinstance(item, int) else sizeof(item)
    size = itemSize * capacity
    if header is not None:
        size += sizeof(header)
    if not skipCounter:
        size += sizeof(QueueView.Counter)

    return size


def as_queue(
    buffer: Buffer,
    item: Type[Structure],
    *,
    offset: int = 0,
    size: Optional[int] = None,
    header: Optional[Type[Structure]] = None,
    skipCounter: bool = False,
) -> QueueView:
    """
    Helper function returning a QueueView pointing at the given buffer.

    Parameters
    ----------
    buffer: Buffer
        Buffer containing the Queue
    item: Structure
        Structure describing a single item
    offset: int, default=0
        Offset in bytes into the buffer the view should start at
    size: int | None, default=None
        Size in bytes of the buffer the view should start.
        If None, the whole buffer minus offset is used.
    header: Structure | None, default=None
        Optional header prefixing the queue
    skipCounter: bool, default=False
        True, if data does not contain a counter

    Returns
    -------
    view: QueueView
        The view describing the queue in the given buffer
    """
    if size is None:
        size = buffer.size_bytes - offset
    # calculate capacity
    if header is not None:
        size -= sizeof(header)
    if not skipCounter:
        size -= sizeof(QueueView.Counter)
    item_size = sizeof(item)
    if size < 0 or size % item_size:
        raise ValueError("The size of the buffer does not match any queue size!")
    capacity = size // item_size
    # create view
    return QueueView(
        buffer.address + offset, item, capacity, skipCounter=skipCounter, header=header
    )


class QueueBuffer(RawBuffer):
    """
    Util class allocating enough memory to hold the given and gives a view to it

    Parameters
    ----------
    item: Structure
        Structure describing a single item
    capacity: int
        Maximum number of items the queue can hold
    header: Structure | None, default=None
        Optional header prefixing the queue
    skipCounter: bool, default=False
        True, if data does not contain a counter
    """

    def __init__(
        self,
        item: Type[Structure],
        capacity: int,
        *,
        header: Optional[Type[Structure]] = None,
        skipCounter: bool = False,
    ) -> None:
        super().__init__(
            queueSize(item, capacity, header=header, skipCounter=skipCounter)
        )
        self._view = QueueView(
            self.address, item, capacity, header=header, skipCounter=skipCounter
        )

    @property
    def capacity(self) -> int:
        """Capacity of the queue"""
        return self.view.capacity

    @property
    def count(self) -> int:
        """Number of items stored in the queue"""
        return self.view.count

    @property
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self.view.item

    @property
    def view(self) -> QueueView:
        """View of the queue stored inside this buffer"""
        return self._view

    def __repr__(self) -> str:
        name, cap = self.item.__name__, self.capacity
        return f"QueueBuffer: {name}[{cap}] ({printSize(self.size_bytes)})"


class QueueTensor(ByteTensor):
    """
    Util class allocating enough memory on the device to hold the given queue

    Parameters
    ----------
    item: Structure
        Structure describing a single item
    capacity: int
        Maximum number of items the queue can hold
    header: Structure | None, default=None
        Optional header prefixing the queue
    skipCounter: bool, default=False
        True, if data does not contain a counter
    """

    def __init__(
        self,
        item: Type[Structure],
        capacity: int,
        *,
        header: Optional[Type[Structure]] = None,
        skipCounter: bool = False,
    ) -> None:
        super().__init__(
            queueSize(item, capacity, header=header, skipCounter=skipCounter)
        )
        self._item = item
        self._capacity = capacity
        self._header = header
        self._hasCounter = not skipCounter

    @property
    def header(self) -> Optional[Structure]:
        """Optional header. None if no header is present"""
        return self._header

    @property
    def hasCounter(self) -> bool:
        """True, if queue has a counter"""
        return self._hasCounter

    @property
    def capacity(self) -> int:
        """Capacity of the queue"""
        return self._capacity

    @property
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self._item

    def __repr__(self) -> str:
        name, cap = self.item.__name__, self.capacity
        return f"QueueBuffer: {name}[{cap}] ({printSize(self.size_bytes)})"


def clearQueue(
    queue: Union[Tensor, QueueTensor],
    *,
    offset: int = 0,
    header: Optional[Type[Structure]] = None,
) -> Command:
    """
    Returns a command to clear the queue, i.e. resets the count thus marking
    any data inside it as garbage.
    """
    if isinstance(queue, QueueTensor):
        if not queue.hasCounter:
            raise ValueError("queue has not counter!")
        header = queue.header
    if header is not None:
        offset += sizeof(header)
    return clearTensor(queue, size=sizeof(QueueView.Counter), offset=offset)


def saveQueue(
    file: Union[str, bytes, PathLike, BinaryIO],
    queue: Union[QueueView, QueueSubView, QueueBuffer],
    *,
    compressed: bool = True,
    skipHeader: bool = False,
) -> None:
    """
    Saves the given queue under the given path or into the given stream.

    Parameters
    ----------
    file: str | bytes | PathLike | BinaryIO
        Either a path-like object specifying the path to the file the result
        is written into or a binary stream the result is written into.
    queue: QueueView | QueueSubView | QueueBuffer
        Buffer to save
    compressed: bool = True
        Whether to compress the data.
    skipHeader: bool = False
        Whether to skip the header during serialization.
        Ignored if there is no header.
    """
    if isinstance(queue, QueueBuffer):
        queue = queue.view
    # save guard for open files
    with ExitStack() as stack:
        # checking for IO is a bit weird so we check for path like and just
        # treat it like a file otherwise
        if isinstance(file, (str, bytes, PathLike)):
            file = stack.enter_context(open(file, "wb"))

        # write header if necessary
        if queue.header is not None and not skipHeader:
            file.write(bytes(queue.header))

        # collect field and save them
        data = {field: queue[field] for field in queue.fields}
        if compressed:
            np.savez_compressed(file, **data)
        else:
            np.savez(file, **data)


def loadQueue(
    file: Union[str, bytes, PathLike, BinaryIO],
    queue: Union[QueueView, QueueSubView, QueueBuffer],
    *,
    skipHeader: bool = False,
    updateCount: bool = True,
) -> None:
    """
    Updates the given queue with the data from the given file or path.

    Parameters
    ----------
    file: str | bytes | PathLike | BinaryIO
        Either a path-like object specifying the path to the file or the file
        itself containing the data to be loaded.
    queue: QueueView | QueueSubview | QueueBuffer
        Queue the data will be loaded into
    compressed: bool = True
        Whether to compress the data.
    updateCount: bool = True
        Whether to update the queue's count. Ignored if queue has no counter.
    """
    if isinstance(queue, QueueBuffer):
        queue = queue.view
    # save guard for open files
    with ExitStack() as stack:
        # checking for IO is a bit weird so we check for path like and just
        # treat it like a file otherwise
        if isinstance(file, (str, bytes, PathLike)):
            file = stack.enter_context(open(file, "rb"))

        # read header if necessary
        if queue.header is not None and not skipHeader:
            data = file.read(sizeof(queue.header))
            # looks quirky, but essentially just a copying new data into header
            pointer(queue.header)[0] = type(queue.header).from_buffer_copy(data)

        # load all fields
        updateQueue(queue, np.load(file), updateCount=updateCount)
