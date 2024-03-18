from typing import Callable, Iterable, Literal, Optional, overload

import hephaistos.pyhephaistos
import numpy.typing
import os

class AccelerationStructure:
    """
    Acceleration Structure used by programs to trace rays against a scene.
    Consists of multiple instances of various Geometries.

    Parameters
    ---------
    instances: GeometryInstance[]
        list of instances the structure consists of
    """

    def __init__(
        self, instances: hephaistos.pyhephaistos.GeometryInstanceVector
    ) -> None:
        """
        Creates an acceleration structure for consumption in shaders from the
        given geometry instances.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the acceleration structure to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the acceleration structure to the program at the given binding
        """
        ...

class AtomicsProperties:
    """
    List of atomic functions a device supports or are enabled
    """

    @property
    def bufferFloat16AtomicAdd(self) -> bool: ...
    @property
    def bufferFloat16AtomicMinMax(self) -> bool: ...
    @property
    def bufferFloat16Atomics(self) -> bool: ...
    @property
    def bufferFloat32AtomicAdd(self) -> bool: ...
    @property
    def bufferFloat32AtomicMinMax(self) -> bool: ...
    @property
    def bufferFloat32Atomics(self) -> bool: ...
    @property
    def bufferFloat64AtomicAdd(self) -> bool: ...
    @property
    def bufferFloat64AtomicMinMax(self) -> bool: ...
    @property
    def bufferFloat64Atomics(self) -> bool: ...
    @property
    def bufferInt64Atomics(self) -> bool: ...
    @property
    def imageFloat32AtomicAdd(self) -> bool: ...
    @property
    def imageFloat32AtomicMinMax(self) -> bool: ...
    @property
    def imageFloat32Atomics(self) -> bool: ...
    @property
    def imageInt64Atomics(self) -> bool: ...
    @property
    def sharedFloat16AtomicAdd(self) -> bool: ...
    @property
    def sharedFloat16AtomicMinMax(self) -> bool: ...
    @property
    def sharedFloat16Atomics(self) -> bool: ...
    @property
    def sharedFloat32AtomicAdd(self) -> bool: ...
    @property
    def sharedFloat32AtomicMinMax(self) -> bool: ...
    @property
    def sharedFloat32Atomics(self) -> bool: ...
    @property
    def sharedFloat64AtomicAdd(self) -> bool: ...
    @property
    def sharedFloat64AtomicMinMax(self) -> bool: ...
    @property
    def sharedFloat64Atomics(self) -> bool: ...
    @property
    def sharedInt64Atomics(self) -> bool: ...

class BindingTraits:
    """
    Properties of binding found in programs
    """

    @property
    def binding(self) -> int:
        """
        Index of the binding
        """
        ...
    @property
    def count(self) -> int:
        """
        Number of elements in binding, i.e. array size
        """
        ...
    @property
    def imageTraits(self) -> Optional[hephaistos.pyhephaistos.ImageBindingTraits]:
        """
        Properties of the image if one is expected
        """
        ...
    @property
    def name(self) -> str:
        """
        Name of the binding. Might be empty.
        """
        ...
    @property
    def type(self) -> hephaistos.pyhephaistos.ParameterType:
        """
        Type of the binding
        """
        ...

class Buffer:
    """
    Base class for all buffers managing memory allocation on the host
    """

    ...

class ByteBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    uint8 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class ByteTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    uint8 and given size. If mapped can be accessed from the host using its memory
    address. Mapping can optionally be requested but will be ignored if the device
    does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class CharBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    int8 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class CharTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    int8 and given size. If mapped can be accessed from the host using its memory
    address. Mapping can optionally be requested but will be ignored if the
    device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class ClearTensorCommand:
    """
    Command for filling a tensor with constant data over a given range
    """

    def __init__(
        self,
        tensor: hephaistos.pyhephaistos.Tensor,
        offset: Optional[int] = None,
        size: Optional[int] = None,
        data: Optional[int] = None,
        unsafe: bool = False,
    ) -> None:
        """
        Creates a command for filling a tensor with constant data over a given
        range. Defaults to zeroing the complete tensor
        """
        ...

class Command:
    """
    Base class for commands running on the device. Execution happens
    asynchronous after being submitted.
    """

class Compiler:
    """
    Compiler for generating SPIR-V byte code used by Programs from shader code
    written in GLSL. Has additional methods to handle includes.
    """

    def __init__(self) -> None: ...
    def addIncludeDir(self, path: os.PathLike) -> None:
        """
        Adds a path to the list of include directories that take part in
        resolving includes.
        """
        ...
    def clearIncludeDir(self) -> None:
        """
        Clears the internal list of include directories
        """
        ...
    def compile(self, code: str, headers: hephaistos.pyhephaistos.HeaderMap) -> bytes:
        """
        Compiles the given GLSL code using the provided header files and returns
        the SPIR-V code as bytes
        """
        ...
    @overload
    def compile(self, code: str) -> bytes:
        """
        Compiles the given GLSL code and returns the SPIR-V code as bytes
        """
        ...
    def popIncludeDir(self) -> None:
        """
        Removes the last added include dir from the internal list
        """
        ...

class DebugMessage:
    """
    Structure describing a debug message including text and some meta information
    """

    @property
    def idName(self) -> str:
        """
        Name of the message type
        """
        ...
    @property
    def idNumber(self) -> int:
        """
        Id of the message type
        """
        ...
    @property
    def message(self) -> str:
        """
        The actual message text
        """
        ...

class DebugMessageSeverityFlagBits:
    """
    Flags indicating debug message severity
    """

    INFO: DebugMessageSeverityFlagBits

    ERROR: DebugMessageSeverityFlagBits

    VERBOSE: DebugMessageSeverityFlagBits

    WARNING: DebugMessageSeverityFlagBits

class Device:
    """
    Handle for a physical device implementing the Vulkan API. Contains basic
    properties of the device.
    """

    @property
    def isDiscrete(self) -> bool:
        """
        True, if the device is a discrete GPU. Can be useful to distinguish from
        integrated ones.
        """
        ...
    @property
    def name(self) -> str:
        """
        Name of the device
        """
        ...

class DispatchCommand:
    """
    Command for executing a program using the given group size
    """

    @property
    def groupCountX(self) -> int:
        """
        Amount of groups in X dimension
        """
        ...
    @groupCountX.setter
    def groupCountX(self, arg: int, /) -> None:
        """
        Amount of groups in X dimension
        """
        ...
    @property
    def groupCountY(self) -> int:
        """
        Amount of groups in Y dimension
        """
        ...
    @groupCountY.setter
    def groupCountY(self, arg: int, /) -> None:
        """
        Amount of groups in Y dimension
        """
        ...
    @property
    def groupCountZ(self) -> int:
        """
        Amount of groups in Z dimension
        """
        ...
    @groupCountZ.setter
    def groupCountZ(self, arg: int, /) -> None:
        """
        Amount of groups in Z dimension
        """
        ...

class DispatchIndirectCommand:
    """
    Command for executing a program using the group size read from the provided tensor at given offset
    """

    @property
    def offset(self) -> int:
        """
        Offset into the Tensor in bytes on where to start reading
        """
        ...
    @offset.setter
    def offset(self, arg: int, /) -> None:
        """
        Offset into the Tensor in bytes on where to start reading
        """
        ...
    @property
    def tensor(self) -> Tensor:
        """
        Tensor from which to read the group size
        """
        ...
    @tensor.setter
    def tensor(self, arg: Tensor, /) -> None:
        """
        Tensor from which to read the group size
        """
        ...

class DoubleBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    double and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class DoubleTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    double and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class FloatBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    float and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class FloatTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    float and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class FlushMemoryCommand:
    """
    Command for flushing memory writes
    """

    ...

class Geometry:
    """
    Underlying structure Acceleration Structures use to trace rays against
    constructed from Meshes.
    """

    def __init__(self) -> None: ...
    @property
    def blas_address(self) -> int:
        """
        device address of the underlying blas
        """
        ...
    @blas_address.setter
    def blas_address(self, arg: int, /) -> None:
        """
        device address of the underlying blas
        """
        ...
    @property
    def indices_address(self) -> int:
        """
        device address of the index buffer, or zero if it was discarded or is
        non existent
        """
        ...
    @indices_address.setter
    def indices_address(self, arg: int, /) -> None:
        """
        device address of the index buffer, or zero if it was discarded or is
        non existent
        """
        ...
    @property
    def vertices_address(self) -> int:
        """
        device address of the vertex buffer or zero if it was discarded
        """
        ...
    @vertices_address.setter
    def vertices_address(self, arg: int, /) -> None:
        """
        device address of the vertex buffer or zero if it was discarded
        """
        ...

class GeometryInstance:
    """
    Building blocks of Acceleration Structures containing a reference to a
    Geometry along a transformation to be applied to the underlying mesh as
    well as additional information.
    """

    def __init__(self) -> None: ...
    @property
    def blas_address(self) -> int:
        """
        device address of the referenced BLAS/geometry
        """
        ...
    @blas_address.setter
    def blas_address(self, arg: int, /) -> None:
        """
        device address of the referenced BLAS/geometry
        """
        ...
    @property
    def customIndex(self) -> int:
        """
        The custom index of this instance available in the shader.
        """
        ...
    @customIndex.setter
    def customIndex(self, arg: int, /) -> None:
        """
        The custom index of this instance available in the shader.
        """
        ...
    @property
    def mask(self) -> int:
        """
        Mask of this instance used for masking ray traces.
        """
        ...
    @mask.setter
    def mask(self, arg: int, /) -> None:
        """
        Mask of this instance used for masking ray traces.
        """
        ...
    @property
    def transform(self) -> numpy.typing.NDArray:
        """
        The transformation applied to the referenced geometry.
        """
        ...
    @transform.setter
    def transform(self, arg: numpy.typing.NDArray, /) -> None:
        """
        The transformation applied to the referenced geometry.
        """
        ...

class GeometryInstanceVector:
    """
    List of GeometryInstance
    """

    def __init__(
        self, arg: Iterable[hephaistos.pyhephaistos.GeometryInstance], /
    ) -> None:
        """
        Construct from an iterable object
        """
        ...
    @overload
    def __init__(self) -> None:
        """
        Default constructor
        """
        ...
    @overload
    def __init__(self, arg: hephaistos.pyhephaistos.GeometryInstanceVector) -> None:
        """
        Copy constructor
        """
        ...
    def append(self, arg: hephaistos.pyhephaistos.GeometryInstance, /) -> None:
        """
        Append `arg` to the end of the list.
        """
        ...
    def clear(self) -> None:
        """
        Remove all items from list.
        """
        ...
    def extend(self, arg: hephaistos.pyhephaistos.GeometryInstanceVector, /) -> None:
        """
        Extend `self` by appending elements from `arg`.
        """
        ...
    def insert(
        self, arg0: int, arg1: hephaistos.pyhephaistos.GeometryInstance, /
    ) -> None:
        """
        Insert object `arg1` before index `arg0`.
        """
        ...
    def pop(self, index: int = -1) -> hephaistos.pyhephaistos.GeometryInstance:
        """
        Remove and return item at `index` (default last).
        """
        ...

class GeometryStore:
    """
    If True, keeps the mesh data on the GPU after building Geometries.
    """

    def __init__(
        self, meshes: hephaistos.pyhephaistos.MeshVector, keepMeshData: bool = True
    ) -> None:
        """
        Creates a geometry store responsible for managing the BLAS/geometries
        used to create and run acceleration structures.
        """
        ...
    def createInstance(self, idx: int) -> hephaistos.pyhephaistos.GeometryInstance:
        """
        Creates a new instance of the specified geometry
        """
        ...
    @property
    def geometries(self) -> hephaistos.pyhephaistos.GeometryVector:
        """
        Returns the list of stored geometries
        """
        ...
    @property
    def size(self) -> int:
        """
        Number of geometries stored
        """
        ...

class GeometryVector:
    """
    List of Geometry
    """

    def __init__(self, arg: Iterable[hephaistos.pyhephaistos.Geometry], /) -> None:
        """
        Construct from an iterable object
        """
        ...
    @overload
    def __init__(self) -> None:
        """
        Default constructor
        """
        ...
    @overload
    def __init__(self, arg: hephaistos.pyhephaistos.GeometryVector) -> None:
        """
        Copy constructor
        """
        ...
    def append(self, arg: hephaistos.pyhephaistos.Geometry, /) -> None:
        """
        Append `arg` to the end of the list.
        """
        ...
    def clear(self) -> None:
        """
        Remove all items from list.
        """
        ...
    def extend(self, arg: hephaistos.pyhephaistos.GeometryVector, /) -> None:
        """
        Extend `self` by appending elements from `arg`.
        """
        ...
    def insert(self, arg0: int, arg1: hephaistos.pyhephaistos.Geometry, /) -> None:
        """
        Insert object `arg1` before index `arg0`.
        """
        ...
    def pop(self, index: int = -1) -> hephaistos.pyhephaistos.Geometry:
        """
        Remove and return item at `index` (default last).
        """
        ...

class HeaderMap:
    """
    Dict mapping filepaths to shader source code. Consumed by Compiler to
    resolve include directives.
    """

    class ItemView: ...
    class KeyView: ...
    class ValueView: ...

    def __init__(self, arg: dict[str, str], /) -> None:
        """
        Construct from a dictionary
        """
        ...
    @overload
    def __init__(self) -> None:
        """
        Default constructor
        """
        ...
    @overload
    def __init__(self, arg: hephaistos.pyhephaistos.HeaderMap) -> None:
        """
        Copy constructor
        """
        ...
    def clear(self) -> None:
        """
        Remove all items
        """
        ...
    def items(self) -> hephaistos.pyhephaistos.HeaderMap.ItemView:
        """
        Returns an iterable view of the map's items.
        """
        ...
    def keys(self) -> hephaistos.pyhephaistos.HeaderMap.KeyView:
        """
        Returns an iterable view of the map's keys.
        """
        ...
    def update(self, arg: hephaistos.pyhephaistos.HeaderMap, /) -> None:
        """
        Update the map with element from `arg`
        """
        ...
    def values(self) -> hephaistos.pyhephaistos.HeaderMap.ValueView:
        """
        Returns an iterable view of the map's values.
        """
        ...

class Image:
    """
    Allocates memory on the device using a memory layout it deems optimal for
    images presenting it inside programs as storage images thus allowing reads
    and writes to it.

    Parameters
    ----------
    format: ImageFormat
        Format of the image
    width: int
        Width of the image in pixels
    height: int, default=1
        Height of the image in pixels
    depth: int, default=1
        Depth of the image in pixels
    """

    def __init__(
        self,
        format: hephaistos.pyhephaistos.ImageFormat,
        width: int,
        height: int = 1,
        depth: int = 1,
    ) -> None: ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str, /
    ) -> None:
        """
        Binds the image to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int, /
    ) -> None:
        """
        Binds the image to the program at the given binding
        """
        ...
    @property
    def depth(self) -> int:
        """
        Depth of the image in pixels
        """
        ...
    @property
    def format(self) -> hephaistos.pyhephaistos.ImageFormat:
        """
        Format of the image
        """
        ...
    @property
    def height(self) -> int:
        """
        Height of the image in pixels
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        Size the image takes in a linear/compact memory layout in bytes. This
        can differ from the actual size the image takes on the device but can
        be useful to allocate buffers for transferring the image.
        """
        ...
    @property
    def width(self) -> int:
        """
        With of the image in pixels
        """
        ...

class ImageBindingTraits:
    """
    Properties a binding expects from a bound image
    """

    @property
    def dims(self) -> int:
        """
        Image dimensions
        """
        ...
    @property
    def format(self) -> hephaistos.pyhephaistos.ImageFormat:
        """
        Expected image format
        """
        ...

class ImageBuffer:
    """
    Utility class allocating memory on the host side in linear memory layout
    allowing easy manipulating of 2D 32bit RGBA image data that can later be
    copied to an image or texture. Provides additional methods for loading and
    saving image data from and to disk in various file formats.

    Parameters
    ----------
    width: int
        Width of the image in pixels
    height: int
        Height of the image in pixels
    """

    def __init__(self, width: int, height: int) -> None: ...
    @property
    def height(self) -> int:
        """
        Height of the image in pixels
        """
        ...
    def loadFile(filename: str) -> hephaistos.pyhephaistos.ImageBuffer:
        """
        Loads the image at the given filepath and returns a new ImageBuffer
        """
        ...
    def loadMemory(data: bytes) -> hephaistos.pyhephaistos.ImageBuffer:
        """
        Loads serialized image data from memory and returns a new ImageBuffer

        Parameters
        ----------
        data: bytes
            Binary data containing the image data
        """
        ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array that allows to manipulate the data handled
        by this ImageBuffer
        """
        ...
    def save(self, filename: str) -> None:
        """
        Saves the image under the given filepath
        """
        ...
    @property
    def width(self) -> int:
        """
        Width of the image in pixels
        """
        ...

class ImageFormat:
    """
    List of supported image formats
    """

    R16G16B16A16_SINT: ImageFormat

    R16G16B16A16_UINT: ImageFormat

    R32G32B32A32_SFLOAT: ImageFormat

    R32G32B32A32_SINT: ImageFormat

    R32G32B32A32_UINT: ImageFormat

    R32G32_SFLOAT: ImageFormat

    R32G32_SINT: ImageFormat

    R32G32_UINT: ImageFormat

    R32_SFLOAT: ImageFormat

    R32_SINT: ImageFormat

    R32_UINT: ImageFormat

    R8G8B8A8_SINT: ImageFormat

    R8G8B8A8_SNORM: ImageFormat

    R8G8B8A8_UINT: ImageFormat

    R8G8B8A8_UNORM: ImageFormat

class IntBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    int32 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class IntTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    int32 and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class LocalSize:
    """
    Description if the local size, i.e. the number and arrangement of threads in
    a single thread group
    """

    def __init__(self) -> None: ...
    @property
    def x(self) -> int:
        """
        Number of threads in X dimension
        """
        ...
    @x.setter
    def x(self, arg: int, /) -> None:
        """
        Number of threads in X dimension
        """
        ...
    @property
    def y(self) -> int:
        """
        Number of threads in Y dimension
        """
        ...
    @y.setter
    def y(self, arg: int, /) -> None:
        """
        Number of threads in Y dimension
        """
        ...
    @property
    def z(self) -> int:
        """
        Number of threads in Z dimension
        """
        ...
    @z.setter
    def z(self, arg: int, /) -> None:
        """
        Number of threads in Z dimension
        """
        ...

class LongBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    int64 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class LongTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    int64 and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class Mesh:
    """
    Representation of a geometric shape consisting of triangles defined by their
    vertices, which can optionally be indexed. Meshes are used to build Geometries.
    """

    def __init__(self) -> None: ...
    @property
    def indices(self) -> numpy.typing.NDArray:
        """
        Optional numpy array holding the indices referencing vertices to create
        triangles.
        """
        ...
    @indices.setter
    def indices(self, arg: numpy.typing.NDArray, /) -> None:
        """
        Optional numpy array holding the indices referencing vertices to create
        triangles.
        """
        ...
    @property
    def vertices(self) -> numpy.typing.NDArray:
        """
        Numpy array holding the vertex data. The first three columns must be the
        x, y and z positions.
        """
        ...
    @vertices.setter
    def vertices(self, arg: numpy.typing.NDArray, /) -> None:
        """
        Numpy array holding the vertex data. The first three columns must be the
        x, y and z positions.
        """
        ...

class MeshVector:
    """
    List of Mesh
    """

    def __init__(self, arg: Iterable[hephaistos.pyhephaistos.Mesh], /) -> None:
        """
        Construct from an iterable object
        """
        ...
    @overload
    def __init__(self) -> None:
        """
        Default constructor
        """
        ...
    @overload
    def __init__(self, arg: hephaistos.pyhephaistos.MeshVector) -> None:
        """
        Copy constructor
        """
        ...
    def append(self, arg: hephaistos.pyhephaistos.Mesh, /) -> None:
        """
        Append `arg` to the end of the list.
        """
        ...
    def clear(self) -> None:
        """
        Remove all items from list.
        """
        ...
    def extend(self, arg: hephaistos.pyhephaistos.MeshVector, /) -> None:
        """
        Extend `self` by appending elements from `arg`.
        """
        ...
    def insert(self, arg0: int, arg1: hephaistos.pyhephaistos.Mesh, /) -> None:
        """
        Insert object `arg1` before index `arg0`.
        """
        ...
    def pop(self, index: int = -1) -> hephaistos.pyhephaistos.Mesh:
        """
        Remove and return item at `index` (default last).
        """
        ...

class ParameterType:
    """
    Type of parameter
    """

    ACCELERATION_STRUCTURE: ParameterType

    COMBINED_IMAGE_SAMPLER: ParameterType

    STORAGE_BUFFER: ParameterType

    STORAGE_IMAGE: ParameterType

    UNIFORM_BUFFER: ParameterType

class Program:
    """
    Encapsulates a shader program enabling introspection into its bindings as
    well as keeping track of the parameters currently bound to them. Execution
    happens trough commands.
    """

    def __init__(self, code: bytes, specialization: bytes) -> None:
        """
        Creates a new program using the shader's byte code.

        Parameters
        ----------
        code: bytes
            Byte code of the program
        specialization: bytes
            Data used for filling in specialization constants
        """
        ...
    @overload
    def __init__(self, code: bytes) -> None:
        """
        Creates a new program using the shader's byte code.

        Parameters
        ----------
        code: bytes
            Byte code of the program
        """
        ...
    def bindParams(*params, **namedparams) -> None:
        """
        Binds the given parameters.
        Positional arguments are bound to the binding of the corresponding
        position. Keyword arguments are matched with the binding of the same
        name.
        """
        ...
    @property
    def bindings(self) -> list[hephaistos.pyhephaistos.BindingTraits]:
        """
        Returns a list of all bindings.
        """
        ...
    def dispatch(
        self, x: int = 1, y: int = 1, z: int = 1
    ) -> hephaistos.pyhephaistos.DispatchCommand:
        """
        Dispatches a program execution with the given amount of workgroups.

        Parameters
        ----------
        x: int, default=1
            Number of groups to dispatch in X dimension
        y: int, default=1
            Number of groups to dispatch in Y dimension
        z: int, default=1
            Number of groups to dispatch in Z dimension
        """
        ...
    def dispatchIndirect(
        self, tensor: hephaistos.pyhephaistos.Tensor, offset: int = 0
    ) -> hephaistos.pyhephaistos.DispatchIndirectCommand:
        """
        Dispatches a program execution using the amount of workgroups stored in
        the given tensor at the given offset. Expects the workgroup size as
        three consecutive unsigned 32 bit integers.

        Parameters
        ----------
        tensor: Tensor
            Tensor from which to read the amount of workgroups
        offset: int, default=0
            Offset at which to start reading
        """
        ...
    def dispatchIndirectPush(
        self, push: bytes, tensor: hephaistos.pyhephaistos.Tensor, offset: int = 0
    ) -> hephaistos.pyhephaistos.DispatchIndirectCommand:
        """
        Dispatches a program execution with the given push data using the amount
        of workgroups stored in the given tensor at the given offset.

        Parameters
        ----------
        push: bytes
            Data pushed to the dispatch as bytes
        tensor: Tensor
            Tensor from which to read the amount of workgroups
        offset: int, default=0
            Offset at which to start reading
        """
        ...
    def dispatchPush(
        self, push: bytes, x: int = 1, y: int = 1, z: int = 1
    ) -> hephaistos.pyhephaistos.DispatchCommand:
        """
        Dispatches a program execution with the given push data and amount of
        workgroups.

        Parameters
        ----------
        push: bytes
            Data pushed to the dispatch as bytes
        x: int, default=1
            Number of groups to dispatch in X dimension
        y: int, default=1
            Number of groups to dispatch in Y dimension
        z: int, default=1
            Number of groups to dispatch in Z dimension
        """
        ...
    def isBindingBound(i: int,/) -> bool:
        """
        Checks wether the i-th binding is currently bound.
        """
        ...
    @overload
    def isBindingBound(name: str,/) -> bool:
        """
        Checks wether the binding specified by its name is currently bound
        """
        ...
    @property
    def localSize(self) -> hephaistos.pyhephaistos.LocalSize:
        """
        Returns the size of the local work group.
        """
        ...

R16G16B16A16_SINT: ImageFormat

R16G16B16A16_UINT: ImageFormat

R32G32B32A32_SFLOAT: ImageFormat

R32G32B32A32_SINT: ImageFormat

R32G32B32A32_UINT: ImageFormat

R32G32_SFLOAT: ImageFormat

R32G32_SINT: ImageFormat

R32G32_UINT: ImageFormat

R32_SFLOAT: ImageFormat

R32_SINT: ImageFormat

R32_UINT: ImageFormat

R8G8B8A8_SINT: ImageFormat

R8G8B8A8_SNORM: ImageFormat

R8G8B8A8_UINT: ImageFormat

R8G8B8A8_UNORM: ImageFormat

class RawBuffer:
    """
    Buffer for allocating a raw chunk of memory on the host accessible via its
    memory address. Useful as a base class providing more complex functionality.

    Parameters
    ----------
    size: int
        size of the buffer in bytes
    """

    def __init__(self, size: int) -> None: ...
    @property
    def address(self) -> int:
        """
        The memory address of the allocated buffer.
        """
        ...
    @property
    def size(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class RetrieveImageCommand:
    """
    Command for copying the image back into the given buffer
    """

    def __init__(
        self, src: hephaistos.pyhephaistos.Image, dst: hephaistos.pyhephaistos.Buffer, /
    ) -> None: ...

class RetrieveTensorCommand:
    """
    Command for copying the src tensor back to the destination buffer
    """

    def __init__(
        self,
        src: hephaistos.pyhephaistos.Tensor,
        dst: hephaistos.pyhephaistos.Buffer,
        bufferOffset: Optional[int] = None,
        tensorOffset: Optional[int] = None,
        size: Optional[int] = None,
        unsafe: bool = False
    ) -> None: ...

class SequenceBuilder:
    """
    Builder class for recording a sequence of commands and subroutines and
    submitting it to the device for execution, which happens asynchronous, i.e.
    submit() returns before the recorded work finishes.
    """

    def And(
        self, subroutine: hephaistos.pyhephaistos.Subroutine
    ) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues the subroutine to run parallel in the current step.
        """
        ...
    @overload
    def And(
        self, cmd: hephaistos.pyhephaistos.Command
    ) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues the command to run parallel in the current step.
        """
        ...
    def AndList(self, list: list) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues each element of the list to run parallel in the current step
        """
        ...
    def NextStep(self) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues a new step. Following calls to And are ensured to run after previous ones finished.
        """
        ...
    def Submit(self) -> hephaistos.pyhephaistos.Submission:
        """
        Submits the recorded steps as a single batch to the GPU.
        """
        ...
    def Then(
        self, subroutine: hephaistos.pyhephaistos.Subroutine
    ) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues a new step to execute after waiting for the previous one to finish.
        """
        ...
    @overload
    def Then(
        self, cmd: hephaistos.pyhephaistos.Command
    ) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues a new step to execute after waiting for the previous one to finish.
        """
        ...
    def WaitFor(
        self, timeline: hephaistos.pyhephaistos.Timeline, value: int
    ) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues the following steps to wait for the given timeline to reach the given value
        """
        ...
    @overload
    def WaitFor(self, value: int) -> hephaistos.pyhephaistos.SequenceBuilder:
        """
        Issues the following steps to wait on the sequence timeline to reach the given value.
        """
        ...
    def __init__(
        self, timeline: hephaistos.pyhephaistos.Timeline, startValue: int
    ) -> None:
        """
        Creates a new SequenceBuilder.

        Parameters
        ----------
        timeline: Timeline
            Timeline to use for orchestrating commands and subroutines
        startValue: int
            Counter value to wait for on the timeline to start with
        """
        ...
    @overload
    def __init__(self, timeline: hephaistos.pyhephaistos.Timeline) -> None:
        """
        Creates a new SequenceBuilder.

        Parameters
        ----------
        timeline: Timeline
            Timeline to use for orchestrating commands and subroutines
        """
        ...
    def printWaitGraph(self) -> str:
        """
        Returns a visualization of the current wait graph in the form:
        (Timeline.ID(WaitValue))* -> (submissions) -> (Timeline.ID(SignalValue)).
        Must be called before Submit().
        """
        ...

class ShortBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    int16 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class ShortTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    int16 and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class StopWatch:
    """
    Allows the measuring of elapsed time between commands execution
    
    """

    def __init__(self) -> None:
        """
        Creates a new stopwatch for measuring elapsed time between commands.
        """
        ...
    def getElapsedTime(self, wait: bool = False) -> list[float]:
        """
        Retrieves the elapsed time between start() and stop() in nanoseconds.
        If wait = True, blocks the caller until all timestamps are available,
        otherwise returns NaN.
        """
        ...
    def reset(self) -> None:
        """
        Resets the stop watch.
        """
        ...
    def start(self) -> hephaistos.pyhephaistos.Command:
        """
        Returns the command to start the stop watch.
        """
        ...
    def stop(self) -> hephaistos.pyhephaistos.Command:
        """
        Returns the command to stop the stop watch. Can be recorded multiple
        times to record up to stopCount stop times.
        """
        ...

class SubgroupProperties:
    """
    List of subgroup properties and supported operations
    """

    @property
    def arithmeticSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_arithmetic
        """
        ...
    @property
    def ballotSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_ballot
        """
        ...
    @property
    def basicSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_basic
        """
        ...
    @property
    def quadSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_quad
        """
        ...
    @property
    def shuffleClusteredSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_clustered
        """
        ...
    @property
    def shuffleRelativeSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_shuffle_relative
        """
        ...
    @property
    def shuffleSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_shuffle
        """
        ...
    @property
    def subgroupSize(self) -> int:
        """
        Threads per subgroup
        """
        ...
    @property
    def voteSupport(self) -> bool:
        """
        Support for GL_KHR_shader_subgroup_vote
        """
        ...

class Submission:
    """
    Submissions are issued after work has been submitted to the device and can
    be used to wait for its completion.
    """

    @property
    def finalStep(self) -> int:
        """
        The value the timeline will reach when the submission finishes.
        """
        ...
    @property
    def forgettable(self) -> bool:
        """
        True, if the Submission can be discarded safely, i.e. fire and forget.
        """
        ...
    @property
    def timeline(self) -> hephaistos.pyhephaistos.Timeline:
        """
        The timeline this submission was issued with.
        """
        ...
    def wait(self) -> None:
        """
        Blocks the caller until the submission finishes.
        """
        ...
    def waitTimeout(self, ns: int) -> bool:
        """
        Blocks the caller until the submission finished or the specified time
        elapsed. Returns True in the first, False in the second case.
        """
        ...

class Subroutine:
    """
    Subroutines are reusable sequences of commands that can be submitted
    multiple times to the device. Recording sequence of commands require non
    negligible CPU time and may be amortized by reusing sequences via Subroutines.
    """

    @property
    def simultaneousUse(self) -> bool:
        """
        True, if the subroutine can be used simultaneous
        """
        ...

class Tensor:
    """
    Base class for all tensors managing memory allocations on the device
    """

    ...

class Texture:
    """
    Allocates memory on the device using a memory layout it deems optimal for
    images and presents it inside programs as texture allowing filtered lookups.
    The filter methods can specified

    Parameters
    ----------
    format: ImageFormat
        Format of the image
    width: int
        Width of the image in pixels
    height: int, default=1
        Height of the image in pixels
    depth: int, default=1
        Depth of the image in pixels
    filter: 'nearest'|'n'|'linear'|'l', default='linear'
        Method used to interpolate between pixels
    unnormalized: bool, default=False
        If True, use coordinates in pixel space rather than normalized ones
        inside programs
    modeU: 'repeat'|'r'|'mirrored repeat'|'mr'|'clamp edge'|'ce'|'mirrored clamp edge'|'mce', default='repeat'
        Method used to handle out of range coordinates in U dimension
    modeV: 'repeat'|'r'|'mirrored repeat'|'mr'|'clamp edge'|'ce'|'mirrored clamp edge'|'mce', default='repeat'
        Method used to handle out of range coordinates in V dimension
    modeW: 'repeat'|'r'|'mirrored repeat'|'mr'|'clamp edge'|'ce'|'mirrored clamp edge'|'mce', default='repeat'
        Method used to handle out of range coordinates in W dimension
    """

    def __init__(
        format: hephaistos.pyhephaistos.ImageFormat,
        width: int,
        height: int = 1,
        depth: int = 1,
        *,
        filter: Literal["nearest", "n", "linear", "l"] = "linear",
        unnormalized: bool = False,
        modeU: Literal[
            "repeat",
            "r",
            "mirrored repeat",
            "mr",
            "clamp edge",
            "ce",
            "mirrored clamp edge",
            "mce",
        ] = "repeat",
        modeV: Literal[
            "repeat",
            "r",
            "mirrored repeat",
            "mr",
            "clamp edge",
            "ce",
            "mirrored clamp edge",
            "mce",
        ] = "repeat",
        modeW: Literal[
            "repeat",
            "r",
            "mirrored repeat",
            "mr",
            "clamp edge",
            "ce",
            "mirrored clamp edge",
            "mce",
        ] = "repeat",
    ) -> None: ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str, /
    ) -> None:
        """
        Binds the texture to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int, /
    ) -> None:
        """
        Binds the texture to the program at the given binding
        """
        ...
    @property
    def depth(self) -> int:
        """
        Depth of the texture in pixels
        """
        ...
    @property
    def format(self) -> hephaistos.pyhephaistos.ImageFormat:
        """
        Format of the texture
        """
        ...
    @property
    def height(self) -> int:
        """
        Height of the texture in pixels
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        Size the texture takes in a linear/compact memory layout in bytes. This
        can differ from the actual size the texture takes on the device but can
        be useful to allocate buffers for transferring the texture.
        """
        ...
    @property
    def width(self) -> int:
        """
        Width of the texture in pixels
        """
        ...

class Timeline:
    """
    Timeline managing the execution of code and commands using an increasing
    internal counter. Both GPU and CPU can wait and/or increase this counter
    thus creating a synchronization between them or among themselves. The
    current value of the counter can be queried allowing an asynchronous method
    of reporting the progress.

    Parameters
    ----------
    value: int, default=0
        Initial value of the internal counter
    """

    def __init__(self, initialValue: int) -> None: ...
    @overload
    def __init__(self) -> None: ...
    @property
    def id(self) -> int:
        """
        Id of this timeline
        """
        ...
    @property
    def value(self) -> int:
        """
        Returns or sets the current value of the timeline. Note that the value
        can only increase. Disobeying this requirement results in undefined
        behavior.
        """
        ...
    @value.setter
    def value(self, arg: int, /) -> None:
        """
        Returns or sets the current value of the timeline. Note that the value
        can only increase. Disobeying this requirement results in undefined
        behavior.
        """
        ...
    def wait(self, value: int) -> None:
        """
        Waits for the timeline to reach the given value.
        """
        ...
    def waitTimeout(self, value: int, timeout: int) -> bool:
        """
        Waits for the timeline to reach the given value for a certain amount.
        Returns True if the value was reached and False if it timed out.

        Parameters
        ----------
        value: int
            Value to wait for
        timeout: int
            Time in nanoseconds to wait. May be rounded to the closest internal
            precision of the device clock.
        """
        ...


class TypeSupport:
    """List of supported extended types, e.g. float64"""

    @property
    def float64(self) -> bool: ...
    @property
    def float16(self) -> bool: ...
    @property
    def int64(self) -> bool: ...
    @property
    def int16(self) -> bool: ...
    @property
    def int8(self) -> bool: ...


class UnsignedIntBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    uint32 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class UnsignedIntTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    uint32 and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class UnsignedLongBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    uint64 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class UnsignedLongTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    uint64 and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class UnsignedShortBuffer:
    """
    Buffer representing memory allocated on the host holding an array of type
    uint16 and given size which can be accessed as numpy array.

    Parameters
    ----------
    size: int
        Number of elements
    """

    def __init__(self, size: int) -> None: ...
    def numpy(self) -> numpy.typing.NDArray:
        """
        Returns a numpy array using this buffer's memory.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this buffer.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the buffer in bytes.
        """
        ...

class UnsignedShortTensor:
    """
    Tensor representing memory allocated on the device holding an array of type
    uint16 and given size. If mapped can be accessed from the host using its
    memory address. Mapping can optionally be requested but will be ignored if
    the device does not support it. Query its support after creation via isMapped.
    """

    def __init__(self, array: numpy.typing.NDArray, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        array: NDArray
            Numpy array containing the data used to fill the tensor.
            Its type must match the tensor's.
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, size: int, mapped: bool = False) -> None:
        """
        Creates a new tensor of given size.

        Parameters
        ----------
        size: int
            Number of elements
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @overload
    def __init__(self, addr: int, n: int, mapped: bool = False) -> None:
        """
        Creates a new tensor and fills it with the provided data

        Parameters
        ----------
        addr: int
            Address of the data used to fill the tensor
        n: int
            Number of bytes to copy from addr
        mapped: bool, default=False
            If True, tries to map memory to host address space
        """
        ...
    @property
    def address(self) -> int:
        """
        The device address of this tensor.
        """
        ...
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: str
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    @overload
    def bindParameter(
        self, program: hephaistos.pyhephaistos.Program, binding: int
    ) -> None:
        """
        Binds the tensor to the program at the given binding
        """
        ...
    def flush(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the host available to the device.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to flush
        size: int|None, default=None
            Number of elements to flush starting at offset.
            If None, flushes all remaining elements.
        """
        ...
    def invalidate(self, offset: int = 0, size: int | None = None, /) -> None:
        """
        Makes writes in mapped memory from the device available to the host.
        Only needed if isNonCoherent is True.

        Parameters
        ----------
        offset: int, default=0
            Offset in amount of elements into mapped memory to invalidate
        size: int | None, default=None
            Number of elements to invalidate starting at offset.
            If None, invalidates all remaining elements.
        """
        ...
    @property
    def isMapped(self) -> bool:
        """
        True, if the underlying memory is writable by the CPU.
        """
        ...
    @property
    def isNonCoherent(self) -> bool:
        """
        Wether calls to flush() and invalidate() are necessary to make changes
        in mapped memory between devices and host available
        """
        ...
    @property
    def memory(self) -> int:
        """
        Mapped memory address of the tensor as seen from the CPU. Zero if not mapped.
        """
        ...
    def retrieve(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Retrieves n elements from the tensor at the given offset and stores it
        in dst.

        Parameters
        ----------
        addr: int
            Address of memory to copy to
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...
    @property
    def size(self) -> int:
        """
        The number of elements in this tensor.
        """
        ...
    @property
    def size_bytes(self) -> int:
        """
        The size of the tensor in bytes.
        """
        ...
    def update(self, addr: int, n: int, offset: int = 0, /) -> None:
        """
        Updates the tensor at the given offset with n elements stored at addr.

        Parameters
        ----------
        addr: int
            Address of memory to copy from
        n: int
            Amount of elements to copy
        offset: int, default=0
            Offset into the tensor in amount of elements where the copy starts
        
        Note
        ----
        This operation is only guaranteed to succeed on mapped tensors.
        """
        ...

class UpdateImageCommand:
    """
    Command for copying data from the given buffer into the image
    """

    def __init__(
        self,
        src: hephaistos.pyhephaistos.Buffer,
        dst: hephaistos.pyhephaistos.Image,
        /,
    ) -> None: ...

class UpdateTensorCommand:
    """
    Command for copying the src buffer into the destination tensor
    """

    def __init__(
        self,
        src: hephaistos.pyhephaistos.Buffer,
        dst: hephaistos.pyhephaistos.Tensor,
        bufferOffset: Optional[int] = None,
        tensorOffset: Optional[int] = None,
        size: Optional[int] = None,
        unsafe: bool = False,
    ) -> None: ...

class UpdateTextureCommand:
    """
    Command for copying data from the given buffer into the texture
    """

    def __init__(
        self,
        src: hephaistos.pyhephaistos.Buffer,
        dst: hephaistos.pyhephaistos.Texture,
        /,
    ) -> None: ...

def beginSequence() -> hephaistos.pyhephaistos.SequenceBuilder:
    """
    Starts a new sequence.
    """
    ...

@overload
def beginSequence(
    timeline: hephaistos.pyhephaistos.Timeline, startValue: int = 0
) -> hephaistos.pyhephaistos.SequenceBuilder:
    """
    Starts a new sequence.
    """
    ...

def clearTensor(
    tensor: hephaistos.pyhephaistos.Tensor,
    offset: Optional[int] = None,
    size: Optional[int] = None,
    data: Optional[int] = None,
    unsafe: bool = False,
) -> hephaistos.pyhephaistos.ClearTensorCommand:
    """
    Creates a command for filling a tensor with constant data over a given range.
    Defaults to zeroing the complete tensor

    Parameters
    ----------
    tensor: Tensor
        Tensor to be modified
    offset: None|int, default=None
        Offset into the Tensor at which to start clearing it.
        Defaults to the start of the Tensor.
    size: None|int, default=None
        Amount of bytes to clear. If None, equals to the range starting at
        offset until the end of the tensor
    data: None|int, default=None
        32 bit integer used to fill the tensor. If None, uses all zeros.
    unsafe: bool, default=False
        Wether to omit barrier ensuring read after write ordering.
    """
    ...

def configureDebug(
    enablePrint: bool = False,
    enableGPUValidation: bool = False,
    enableSynchronizationValidation: bool = False,
    enableThreadSafetyValidation: bool = False,
    enableAPIValidation: bool = False,
    callback: Callable[[hephaistos.pyhephaistos.DebugMessage], None] = lambda msg: None,
) -> None:
    """
    Configures the current debug state. Must be called before any other library
    calls except isVulkanAvailable() and isDebugAvailable().

    Parameters
    ----------
    enablePrint: bool, default=False
        Enables the usage of GL_EXT_debug_printf
    enableGPUValidation: bool, default=False
        Enable GPU assisted validation
    enableSynchronizationValidation: bool, default=False
        Enable synchronization validation between resources
    enableThreadSafetyValidation: bool, default=False
        Enable thread safety validation
    enableAPIValidation: bool, default=False
        Enables validation of the Vulkan API usage
    callback: (DebugMessage) -> None, default=print(message)
        Callback called for each message. Prints the message by default.

    Note
    ----
    Debugging requires the Vulkan Validation Layers to be installed.
    """
    ...

def createSubroutine(
    commands: list, simultaneous: bool = False
) -> hephaistos.pyhephaistos.Subroutine:
    """
    creates a subroutine from the list of commands

    Parameters
    ----------
    commands: Command[]
        Sequence of commands the Subroutine consists of
    simultaneous: bool, default=False
        True, if the subroutine can be submitted while a previous submission
        has not yet finished. Disobeying this requirement results in undefined
        behavior.
    """
    ...

def enableAtomics(flags: set, force: bool = False) -> None:
    """
    Enables the atomic features contained in the given set by their name. Set
    force=True if an existing context should be destroyed.
    """
    ...

def enableRaytracing(force: bool = False) -> None:
    """
    Enables ray tracing. (Lazy) context creation fails if not supported. Set
    force=True if an existing context should be destroyed.
    """
    ...

def enumerateDevices() -> list[hephaistos.pyhephaistos.Device]:
    """
    Returns a list of all supported installed devices.
    """
    ...

def execute(sub: hephaistos.pyhephaistos.Subroutine) -> None:
    """
    Runs the given subroutine synchronous.
    """
    ...

@overload
def execute(cmd: hephaistos.pyhephaistos.Command) -> None:
    """
    Runs the given command synchronous.
    """
    ...

def executeList(list: list) -> None:
    """
    Runs the given of list commands synchronous
    """
    ...

def flushMemory() -> hephaistos.pyhephaistos.FlushMemoryCommand:
    """
    Returns a command for flushing memory writes.
    """
    ...

def getAtomicsProperties(id: int) -> hephaistos.pyhephaistos.AtomicsProperties:
    """
    Returns the atomic capabilities of the device given by its id
    """
    ...

def getCurrentDevice() -> hephaistos.pyhephaistos.Device:
    """
    Returns the currently active device. Note that this may initialize the context.
    """
    ...

def getElementSize(format: hephaistos.pyhephaistos.ImageFormat) -> int:
    """
    Returns the size of a single channel in bytes
    """
    ...

def getEnabledAtomics() -> hephaistos.pyhephaistos.AtomicsProperties:
    """
    Returns the in the current context enabled atomic features
    """
    ...

def getSubgroupProperties(arg: int, /) -> hephaistos.pyhephaistos.SubgroupProperties:
    """
    Returns the properties specific to subgroups (waves).
    """
    ...

@overload
def getSubgroupProperties() -> hephaistos.pyhephaistos.SubgroupProperties:
    """
    Returns the properties specific to subgroups (waves).
    """
    ...

def getSupportedTypes(id: Optional[int], /) -> hephaistos.pyhephaistos.TypeSupport:
    """
    Queries the supported extended types

    Parameters
    ----------
    id: int | None, default=None
        Id of the device to query. If None, uses the one the current context
        was created on
    """
    ...

def isDebugAvailable() -> bool:
    """
    Returns True if debugging is supported on this system.
    This requires the Vulkan Validation Layers to be installed.
    """
    ...

def isDeviceSuitable(arg: int, /) -> bool:
    """
    Returns True if the device given by its id supports all enabled extensions
    """
    ...

def isRaytracingEnabled() -> bool:
    """
    Checks wether ray tracing was enabled. Note that this creates the context.
    """
    ...

def isRaytracingSupported(id: Optional[int] = None) -> bool:
    """
    Checks wether any or the given device supports ray tracing.
    """
    ...

def isVulkanAvailable() -> bool:
    """
    Returns True if Vulkan is available on this system.
    """
    ...

def requireTypes(types: set, force: bool = False, /) -> None:
    """
    Forces the given types as specified by in the set by their names (f64, f16,
    i64, i16, i8), effectively marking devices, which do not support them as
    unsupported. Set force=True if an existing context should be destroyed.
    """

def retrieveImage(
    src: hephaistos.pyhephaistos.Image, dst: hephaistos.pyhephaistos.Buffer
) -> hephaistos.pyhephaistos.RetrieveImageCommand:
    """
    Creates a command for copying the image back into the given buffer.

    Parameters
    ----------
    src: Image
        Source image
    dst: Buffer
        Destination buffer
    """
    ...

def retrieveTensor(
    src: hephaistos.pyhephaistos.Tensor,
    dst: hephaistos.pyhephaistos.Buffer,
    bufferOffset: Optional[int] = None,
    tensorOffset: Optional[int] = None,
    size: Optional[int] = None,
    unsafe: bool = False,
) -> hephaistos.pyhephaistos.RetrieveTensorCommand:
    """
    Creates a command for copying the src tensor back to the destination buffer

    Parameters
    ----------
    src: Tensor
        Source tensor
    dst: Buffer
        Destination buffer
    bufferOffset: None|int, default=None
        Optional offset into the buffer in bytes
    tensorOffset: None|int, default=None
        Optional offset into the tensor in bytes
    size: None|int, default=None
        Amount of data to copy in bytes. If None, equals to the complete buffer
    unsafe: bool, default=False
        Wether to omit barrier ensuring read after write ordering
    """
    ...

def selectDevice(id: int, force: bool = False) -> None:
    """
    Sets the device on which the context will be initialized. Set force=True if
    an existing context should be destroyed.
    """
    ...

def suitableDeviceAvailable() -> bool:
    """
    Returns True, if there is a device available supporting all enabled extensions
    """
    ...

def updateImage(
    src: hephaistos.pyhephaistos.Buffer, dst: hephaistos.pyhephaistos.Image
) -> hephaistos.pyhephaistos.UpdateImageCommand:
    """
    Creates a command for copying data from the given buffer into the image.

    Parameters
    ----------
    src: Buffer
        Source buffer
    dst: Image
        Destination image
    """
    ...

def updateTensor(
    src: hephaistos.pyhephaistos.Buffer,
    dst: hephaistos.pyhephaistos.Tensor,
    bufferOffset: Optional[int] = None,
    tensorOffset: Optional[int] = None,
    size: Optional[int] = None,
    unsafe: bool = False,
) -> hephaistos.pyhephaistos.UpdateTensorCommand:
    """
    Creates a command for copying the src buffer into the destination tensor

    Parameters
    ----------
    src: Buffer
        Source Buffer
    dst: Tensor
        Destination Tensor
    bufferOffset: None|int, default=None
        Optional offset into the buffer in bytes
    tensorOffset: None|int, default=None
        Optional offset into the tensor in bytes
    size: None|int, default=None
        Amount of data to copy in bytes. If None, equals to the complete buffer
    unsafe: bool, default=False
        Wether to omit barrier ensuring read after write ordering
    """
    ...

def updateTexture(
    src: hephaistos.pyhephaistos.Buffer, dst: hephaistos.pyhephaistos.Texture
) -> hephaistos.pyhephaistos.UpdateTextureCommand:
    """
    Creates a command for copying data from the given buffer into the texture.

    Parameters
    ----------
    src: Buffer
        Source buffer
    dst: Texture
        Destination texture
    """
    ...
