from setuptools import setup, find_packages

setup(
    name='gentleboost',
    version='0.1.0',
    description='GPU-accelerated GentleBoost implementation using cuML and RAPIDS',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'cuml',
        'cupy-cuda12x',  # Replace '12x' with your CUDA version
        'scikit-learn',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
