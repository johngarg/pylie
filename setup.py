from setuptools import setup, find_packages

with open('README.rst') as f:
        readme = f.read()

setup(name = 'pylie',
      version = '1.0.0', 
      description = 'Python port of group theory functionality of SusyNo',
      long_description = readme,
      packages=find_packages(exclude=['docs'])
      )
