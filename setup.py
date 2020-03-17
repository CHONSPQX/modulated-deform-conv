import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, sources,includes):

    return CUDAExtension(
        name='{}'.format(name),
        sources=[p for p in sources],
        include_dirs=[i for i in includes],
        extra_compile_args={
            'cxx':  [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]})
#-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D_GLIBCXX_USE_CXX11_ABI=1
sources=[]
sources.extend(glob.glob('src/*.cu'))
sources.extend(glob.glob('src/*.cpp'))

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='modulated_deform_conv',
    version='1.0.2',
    author='qiaoxin',
    author_email='qiaoxin182@gmail.com',
    url='https://www.github.com',
    description="cuda implementation of deformable conv2d, modulated deformable conv2d,"
                "deformable conv3d, modulated deformable conv3d",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=[
        make_cuda_ext(name='MDCONV_CUDA',
                      sources=sources,
                      includes=['src'])
    ],
    py_modules=['modulated_deform_conv'],
    classifiers=(
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Operating System :: POSIX :: Linux',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ),
    install_requires=['torch>=1.3'],
    keywords=["pytorch", "cuda", "deform"],
    cmdclass={'build_ext': BuildExtension}, zip_safe=False)
