#!/usr/bin/env python

import os
import setuptools

def get_long_description():
    filename = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(filename) as f:
        return f.read()

setuptools.setup(name='tsalib',
      version='0.2.1',
      description="TSAlib: Support for Named Tensor Shapes",
      long_description=get_long_description(),
      long_description_content_type="text/markdown",
      author='Nishant Sinha',
      author_email='nishantsinha@acm.org',
      url='https://github.com/ofnote/tsalib',
      license='Apache 2.0',
      platforms=['POSIX'],
      packages=setuptools.find_packages(),
      #entry_points={},
      classifiers=[
          'Environment :: Console',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
          ],
      setup_requires=['sympy'],
      install_requires=['sympy'],
      )