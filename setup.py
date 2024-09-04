"""
coastal-fieldwork

Author: Jakob C. Christiaanse
"""
from setuptools import setup

with open('README.md', mode='r') as f:
    long_description = f.read()

setup(
    name='coastal-fieldwork',
    version='1.0',
    author='Jakob C. Christiaanse',
    author_email='J.C.Christiaanse@tudelft.nl',
    description='Python based readout and processing of coastal fieldwork data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['src'],
    license='Apache-2.0',
    keywords=['Coastal Engineering', 'Field Work', 'Data Processing'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'pandas>=2.0.3',
        'scipy',
        'pyrsktools',
        'tqdm',
        'pyproj',
        'matplotlib'],
    python_requires='>=3.7'
)
