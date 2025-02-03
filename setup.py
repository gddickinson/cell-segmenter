from setuptools import setup, find_packages

setup(
    name="cell_segmenter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'PyQt6',
        'pyqtgraph',
        'scikit-image',
        'scipy',
        'torch',
        'tifffile'
    ]
)