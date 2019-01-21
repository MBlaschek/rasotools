from setuptools import setup

setup(name='rasotools',
      version='0.2',
      description='radiosonde tools',
      url='https://github.com/MBlaschek/rasotools',
      author='MB',
      author_email='michael.blaschek@univie.ac.at',
      license='UNIVIE GNU GPL',
      packages=['rasotools'],
      install_requires=['numpy', 'pandas', 'netCDF4', 'xarray', 'matplotlib', 'numba'],
      zip_safe=False)