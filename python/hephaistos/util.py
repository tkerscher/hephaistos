import ctypes
import subprocess
import tempfile
import os.path
from typing import Any
from .pyhephaistos import RawBuffer, ByteTensor, Program

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

def isCompilerAvailable() -> bool:
    """
    Tests wether the glsl compiler is available.
    """
    return subprocess.run(
        "glslangvalidator --version",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ).returncode == 0

def compileShader(code: str) -> bytes:
    """
    Calls the glsl compiler to process the given code and returns the created bytecode.
    """
    with tempfile.TemporaryDirectory() as tmpDir:
        path = os.path.join(tmpDir, "shader")
        result = subprocess.run(
            f"glslangvalidator --target-env vulkan1.2 --stdin --quiet -g0 -V -S comp -o {path}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            input=code, text=True)
        
        if result.returncode != 0:
            raise RuntimeError("Error while compiling code: " + result)
        
        with open(path, mode="rb") as file:
            return file.read()
