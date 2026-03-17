from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import os
import os.path
import shutil
import subprocess
import sys


class BuildAvocadoExt(_build_ext):
    """Builds AVOCADO before our module."""

    def run(self):
        build_dir = os.path.abspath('build/AVOCADO')

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        cmake_exe = shutil.which('cmake') or os.path.join(sys.prefix, 'Scripts', 'cmake.exe')

        configure_cmd = [cmake_exe, '../..']
        if os.name != 'nt':
            configure_cmd.append('-DCMAKE_CXX_FLAGS=-fPIC')

        subprocess.check_call(configure_cmd, cwd=build_dir)

        build_cmd = [cmake_exe, '--build', '.']
        if os.name == 'nt':
            build_cmd.extend(['--config', 'Release'])
        subprocess.check_call(build_cmd, cwd=build_dir)

        _build_ext.run(self)


library_dirs = ['build/AVOCADO/src']
extra_compile_args = []
if os.name != 'nt':
    extra_compile_args.append('-fPIC')
else:
    library_dirs.insert(0, 'build/AVOCADO/src/Release')


extensions = [
    Extension('avocado', ['src/*.pyx'],
              include_dirs=['src'],
              libraries=['AVOCADO'],
              library_dirs=library_dirs,
              extra_compile_args=extra_compile_args),
]

setup(
    name="pyavocado",
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': BuildAvocadoExt},
    classifiers=[
        # 'Development Status :: 5 - Production/Stable',
        # 'Intended Audience :: Developers',
        # 'Intended Audience :: Education',
        # 'Intended Audience :: Information Technology',
        # 'Operating System :: OS Independent',
        # 'Programming Language :: Python',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Cython',
        # 'Topic :: Games/Entertainment :: Simulation',
        # 'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
