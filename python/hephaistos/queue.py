from __future__ import annotations

import numpy as np
import warnings

from ctypes import Structure, c_uint32, sizeof
from hephaistos import Buffer, ByteTensor, Command, RawBuffer, Tensor, clearTensor
from hephaistos.util import printSize
from numpy import ndarray
from numpy.ctypeslib import as_array

from numpy.typing import NDArray
from typing import Any, Dict, Optional, Set, Type, Union, overload


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
    skipHeader: bool, default=False
        True, if data does not contain a queue header
    """

    class Header(Structure):
        """Structure describing the queue header"""

        _fields_ = [
            ("count", c_uint32),
        ]

    def __init__(
        self, data: int, item: Type[Structure], capacity: int, *, skipHeader: int
    ) -> None:
        # store item type
        self._item = item
        self._capacity = capacity
        # read header
        if not skipHeader:
            self._header = self.Header.from_address(data)
            data += sizeof(self.Header)
        else:
            self._header = None

        # create SoA and use it to create data access
        class SoA(Structure):
            _fields_ = [
                (name, t._type_ * capacity * t._length_)  # handle arrays
                if hasattr(t, "_length_")  # check if array
                else (name, t * capacity)  # handle scalar
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
    def hasHeader(self) -> bool:
        """True, if queue has a header"""
        return self._header is not None

    @property
    def count(self) -> int:
        """Number of items in queue. Equal to capacity if header was skipped"""
        if self._header is not None:
            return self._header.count
        else:
            return self.capacity

    @count.setter
    def count(self, value: int) -> None:
        if self._header is not None:
            self._header.count = value
        else:
            raise ValueError("The queue has no header and thus no count can be set!")

    @property
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self._item

    @overload
    def save(self, file: None, *, compressed: bool) -> Dict[str, NDArray]:
        ...

    @overload
    def save(self, file: str, *, compressed: bool) -> None:
        ...

    def save(
        self, file: Optional[str] = None, *, compressed: bool = True
    ) -> Union[None, Dict[str, NDArray]]:
        """
        Serializes the complete queue and either returns it as a dictionary
        containing independent copies of the fields as numpy arrays or saves
        them to disk at the given path.

        Parameters
        ----------
        file: str|None, default=None
            Path to where the queue is to be saved. If None, returns a dict
            instead.
        compressed: bool, default=True
            True, if the file should be compressed. Ignored if no path is given

        Returns
        -------
        fields: { field: array } | None
            If no path is given, returns a dictionary with numpy arrays
            containing copies of each field, None otherwise.
        """
        if file is None:
            return {field: self[field][: self.count].copy() for field in self.fields}
        else:
            d = {field: self[field][: self.count] for field in self.fields}
            if compressed:
                np.savez_compressed(file, **d)
            else:
                np.savez(file, **d)

    def load(self, data: str | Dict[str, NDArray]) -> None:
        """
        Loads the given data into this queue and updates its count.
        Data can either be the path to a file containing the data to be load, or
        a dictionary containing a numpy arrays for each field.

        Parameters
        ----------
        data: str|{ field: array }
            Data either as path to file to load, or dictionary mapping field
            names to arrays containing the data.
        """
        counts = []

        # open file if provided
        file = None
        if type(data) is not dict:
            file = np.load(data)
            data = file

        # load all fields
        for field, arr in data.items():
            # check if field exists
            if field not in self.fields:
                warnings.warn(f"Skipping unknown field {field}")
                continue
            # check field size
            if len(arr) > self.capacity:
                warnings.warn(f'Field "{field}" truncated to queue\'s capacity')
            # store data
            n = min(self.capacity, len(arr))
            self[field][:n] = arr[:n]
            counts.append(self.capacity if len(arr) > self.capacity else len(arr))

        # close file
        if file is not None:
            file.close()

        # check all have the same
        if len(counts) > 0 and not all(c == counts[0] for c in counts):
            warnings.warn("Not all fields have the same length!")
        # set count if there is an header
        if self.hasHeader:
            self.count = min(counts)

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
    def item(self) -> Type[Structure]:
        """Structure describing the items of the queue"""
        return self._orig.item

    @overload
    def save(self, file: None, *, compressed: bool) -> Dict[str, NDArray]:
        ...

    @overload
    def save(self, file: str, *, compressed: bool) -> None:
        ...

    def save(
        self, file: Optional[str] = None, *, compressed: bool = True
    ) -> Union[None, Dict[str, NDArray]]:
        """
        Serializes the complete queue and either returns it as a dictionary
        containing independent copies of the fields as numpy arrays or saves
        them to disk at the given path.

        Parameters
        ----------
        file: str|None, default=None
            Path to where the queue is to be saved. If None, returns a dict
            instead.
        compressed: bool, default=True
            True, if the file should be compressed. Ignored if no path is given

        Returns
        -------
        fields: { field: array } | None
            If no path is given, returns a dictionary with numpy arrays
            containing copies of each field, None otherwise.
        """
        if file is None:
            return {field: self[field].copy() for field in self.fields}
        else:
            d = {field: self[field] for field in self.fields}
            if compressed:
                np.savez_compressed(file, **d)
            else:
                np.savez(file, **d)

    def load(self, data: str | Dict[str, NDArray]) -> None:
        """
        Loads the given data into this queue and updates its count.
        Data can either be the path to a file containing the data to be load, or
        a dictionary containing a numpy arrays for each field.

        Parameters
        ----------
        data: str|{ field: array }
            Data either as path to file to load, or dictionary mapping field
            names to arrays containing the data.
        """
        counts = []

        # open file if provided
        file = None
        if type(data) is not dict:
            file = np.load(data)
            data = file

        # load all fields
        for field, arr in data.items():
            # check if field exists
            if field not in self.fields:
                warnings.warn(f"Skipping unknown field {field}")
                continue
            # check field size
            if len(arr) > len(self):
                warnings.warn(f'Field "{field}" truncated to view\'s size')
            # store data
            n = min(len(self), len(arr))
            self[field][:n] = arr[:n]
            counts.append(len(self) if len(arr) > len(self) else len(arr))

        # close file
        if file is not None:
            file.close()

        # check all have the same
        if len(counts) > 0 and not all(c == counts[0] for c in counts):
            warnings.warn("Not all fields have the same length!")

    def __repr__(self) -> str:
        return f"QueueSubView: {self.item.__name__}[{self._count}]"


def queueSize(
    item: Union[Type[Structure], int], capacity: int, *, skipHeader: bool = False
) -> int:
    """
    Calculates the required size to store a queue of given size and item type

    Parameters
    ----------
    item: Structure | int
        Either a Structure describing a single item or the size of it
    capacity: int
        Maximum number of items the queue can hold
    skipHeader: bool, default=False
        True, if the queue should not contain a header
    """
    itemSize = item if isinstance(item, int) else sizeof(item)
    if skipHeader:
        return itemSize * capacity
    else:
        return sizeof(QueueView.Header) + itemSize * capacity


def as_queue(
    buffer: Buffer,
    item: Type[Structure],
    *,
    offset: int = 0,
    size: Optional[int] = None,
    skipHeader: bool = False,
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
    skipHeader: bool, default=False
        True, if data does not contain a queue header

    Returns
    -------
    view: QueueView
        The view describing the queue in the given buffer
    """
    if size is None:
        size = buffer.size_bytes - offset
    # calculate capacity
    if not skipHeader:
        size -= sizeof(QueueView.Header)
    item_size = sizeof(item)
    if size < 0 or size % item_size:
        raise ValueError("The size of the buffer does not match any queue size!")
    capacity = size // item_size
    # create view
    return QueueView(buffer.address + offset, item, capacity, skipHeader=skipHeader)


class QueueBuffer(RawBuffer):
    """
    Util class allocating enough memory to hold the given and gives a view to it

    Parameters
    ----------
    item: Structure
        Structure describing a single item
    capacity: int
        Maximum number of items the queue can hold
    skipHeader: bool, default=False
        True, if the queue should not contain a header
    """

    def __init__(
        self, item: Type[Structure], capacity: int, *, skipHeader: bool = False
    ) -> None:
        super().__init__(queueSize(item, capacity, skipHeader=skipHeader))
        self._view = QueueView(self.address, item, capacity, skipHeader=skipHeader)

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
    skipHeader: bool, default=False
        True, if the queue should not contain a header
    """

    def __init__(
        self, item: Type[Structure], capacity: int, *, skipHeader: bool = False
    ) -> None:
        super().__init__(queueSize(item, capacity, skipHeader=skipHeader))
        self._item = item
        self._capacity = capacity
        self._skipHeader = skipHeader

    @property
    def hasHeader(self) -> bool:
        """True if the queue has a header"""
        return not self._skipHeader

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


def clearQueue(queue: Tensor) -> Command:
    """
    Returns a command to clear the queue, i.e. resets the count thus marking
    any data inside it as garbage.
    """
    return clearTensor(queue, size=sizeof(QueueView.Header))
