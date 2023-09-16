import sys, re, os

try:
    from skbuild import setup
    import nanobind
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

setup(
    name="hephaistos",
    version="0.3.0",
    author="Kerscher Tobias",
    author_email="88444139+tkerscher@users.noreply.github.com",
    description="GPU computing framework using the Vulkan API",
    url="https://github.com/tkerscher/hephaistos",
    license="MIT",
    packages=['hephaistos'],
    package_dir={'': 'python'},
    cmake_install_dir="python/hephaistos",
    include_package_data=True,
    python_requires=">=3.8"
)
