from setuptools import setup, find_packages


from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "kohonen/README.md").read_text() 

setup(
    name='kohonen',
    version='0.1.0',
    author='Dimas Pradana, co-authored by a lot of LLM',
    author_email='dimas.putra@bmkg.go.id',
    description='A PyTorch-based Self-Organizing Map (SOM) library accelerated with Triton kernels.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Arcturion/Kohonen-from-Scratch',
    packages=find_packages(where=".", include=['kohonen', 'kohonen.*']),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0', # Requires PyTorch 2.0+ for good Triton support
    ],
    keywords='som, self-organizing map, kohonen map, triton, pytorch, gpu, machine learning, deep learning',
    project_urls={
        'Bug Reports': 'https://github.com/Arcturion/Kohonen-from-Scratch/issues',
        'Source': 'https://github.com/Arcturion/Kohonen-from-Scratch',
    },
)
