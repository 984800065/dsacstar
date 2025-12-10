from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import subprocess

# 通过 pkg-config 获取系统 OpenCV4 的编译和链接参数
def pkgconfig_flags(mod):
    flags = subprocess.check_output(['pkg-config', '--cflags', '--libs', mod]).decode().strip().split()
    include_dirs = [f[2:] for f in flags if f.startswith('-I')]
    library_dirs = [f[2:] for f in flags if f.startswith('-L')]
    libraries = [f[2:] for f in flags if f.startswith('-l')]
    return include_dirs, library_dirs, libraries

opencv_inc_dirs, opencv_lib_dirs, opencv_libs = pkgconfig_flags('opencv4')
print(opencv_inc_dirs, opencv_lib_dirs, opencv_libs)
setup(
    name='dsacstar',
    ext_modules=[
        CppExtension(
            name='dsacstar',
            sources=['dsacstar.cpp', 'thread_rand.cpp'],
            include_dirs=opencv_inc_dirs,
            library_dirs=opencv_lib_dirs,
            libraries=opencv_libs,
            extra_compile_args=['-fopenmp', '-std=c++17']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
