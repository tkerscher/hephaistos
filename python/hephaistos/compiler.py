from abc import ABC, abstractmethod
import subprocess
import tempfile
import os.path

class ShaderCompiler(ABC):
    """
    Class for abstracting shader compilers.
    """
    @property
    @abstractmethod
    def isAvailable(self) -> bool:
        """True, if the compiler is available on this system."""
        pass

    @abstractmethod
    def compileFile(self, path) -> bytes:
        """
        Compiles the shader stored at the given filepath and returns the
        compiled byte code.
        """
        pass

    @abstractmethod
    def compileSource(self, code: str) -> bytes:
        """
        Compiles the given source code and returns the compiled byte code.
        """
        pass

class GLSLangCompiler(ShaderCompiler):
    """
    Wrapper for the GLSLangValidator compiler.
    """
    
    def __init__(self, exec: str = "glslangvalidator"):
        """
        Creates a new GLSLangValidator wrapper using the given executable.
        """
        super().__init__()
        self.exec = exec
        #check if compiler is available
        result = subprocess.run(f"{self.exec} --version", shell=True,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        self._available = result.returncode == 0
    
    @property
    def isAvailable(self) -> bool:
        return self._available
    
    def compileSource(self, code: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpDir:
            path = os.path.join(tmpDir, "shader")
            result = subprocess.run(
                f"{self.exec} --target-env vulkan1.2 --stdin --quiet -V -S comp -o {path}",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                input=code, text=True)
            
            if result.returncode != 0:
                raise RuntimeError("Error while compiling code: " + result)
            
            with open(path, mode="rb") as file:
                return file.read()
    
    def compileFile(self, filepath: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpDir:
            path = os.path.join(tmpDir, "shader")
            result = subprocess.run(
                f"glslangvalidator --target-env vulkan1.2 --quiet -V -S comp -o {path} {filepath}",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                raise RuntimeError("Error while compiling code: " + result)
            
            with open(path, mode="rb") as file:
                return file.read()

class GLSLCCompiler(ShaderCompiler):
    """
    Wrapper for the GLSLC compiler using the given executable.
    """

    def __init__(self, exec: str = "glslc"):
        """
        Creates a new GLSLC wrapper using the given executable.
        """
        super().__init__()
        self.exec = exec
        #check if the compiler is available
        result = subprocess.run(f"{self.exec} --version", shell=True,
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        self._available = result.returncode == 0
    
    @property
    def isAvailable(self) -> bool:
        return self._available

    def compileSource(self, code: str) -> bytes:
        result = subprocess.run(f"{self.exec} --target-env=vulkan1.2 -fshader-stage=comp -o - -",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            input=code.encode("ascii"))
        
        if result.returncode != 0:
            raise RuntimeError("Error while compiling code: " + result.stderr.decode("ascii"))

        return result.stdout
    
    def compileFile(self, filepath: str) -> bytes:
        result = subprocess.run(f"{self.exec} --target-env=vulkan1.2 -fshader-stage=comp -o - {filepath}",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            raise RuntimeError("Error while compiling code: " + result.stderr.decode("ascii"))

        return result.stdout

#list of known compilers for automatic discovery with descending priority
_compilers = [
    GLSLCCompiler(),
    GLSLangCompiler()
]

def isCompilerAvailable() -> bool:
    """
    Returns true, if at least one shader compiler has been found.
    """
    return any(c.isAvailable for c in _compilers)

def compileFile(filepath: str, compiler: ShaderCompiler = None) -> bytes:
    """
    Compiles the shader stored at the given filepath using the given compiler
    and returns the compiled byte code. If no compiler is specified an
    available one is chosen from a list.
    """
    if compiler is None:
        compiler = next((c for c in _compilers if c.isAvailable), None)
    if compiler is None:
        raise RuntimeError("No suitable compiler found!")
    
    return compiler.compileFile(filepath)

def compileSource(code: str, compiler: ShaderCompiler = None) -> bytes:
    """
    Compiles the source code using the given compiler
    and returns the compiled byte code. If no compiler is specified an
    available one is chosen from a list.
    """
    if compiler is None:
        compiler = next((c for c in _compilers if c.isAvailable), None)
    if compiler is None:
        raise RuntimeError("No suitable compiler found!")
    
    return compiler.compileSource(code)
