from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "kohonen/README.md").read_text() # Use the library's README

setup(
    name='kohonen',
    version='0.1.0', # Corresponds to __version__ in __init__.py
    author='Jules (AI Agent)', # Or your name/organization
    author_email='your_email@example.com', # Or your email
    description='A PyTorch-based Self-Organizing Map (SOM) library accelerated with Triton kernels.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https_your_repo_url_here', # URL to the repository
    packages=find_packages(where=".", include=['kohonen', 'kohonen.*']),
    # We specify 'kohonen' as a top-level package.
    # If you had other top-level packages, you'd list them here or adjust find_packages.
    # Example: packages=['kohonen'] if it's the only top-level package.
    # find_packages() is generally good for discovering them.
    classifiers=[
        'Development Status :: 3 - Alpha', # Or Beta, Production/Stable
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Assuming MIT, change if different
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent', # Triton has specific OS/GPU requirements
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0', # Requires PyTorch 2.0+ for good Triton support
        # Triton is often a peer dependency or installed with PyTorch,
        # but can be listed if a specific version is needed: 'triton'
    ],
    keywords='som, self-organizing map, kohonen map, triton, pytorch, gpu, machine learning, deep learning',
    project_urls={ # Optional
        'Bug Reports': 'https_your_repo_url_here/issues',
        'Source': 'https_your_repo_url_here',
    },
)
