from setuptools import setup, find_packages

setup(
    name='rasotools',
    version='0.2.0',
    author='Ulrich Voggenberger',
    author_email='ulrich.voggenberger@univie.ac.at',
    description='Tools for radiosonde data processing and analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://pypi.python.org/pypi/Rasotools/',  # optional
    packages=find_packages(include=['rasotools', 'rasotools.*']),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'rsjupyter=rasotools.some_module:main_function',
            'rsncnfo=rasotools.other_module:main_function',
        ],
    },
    install_requires=[
        # list your Python dependencies here or use requirements.txt
        'numpy',
        'numexpr >= 2.4',
        'xarray',
        'matplotlib',
        'netCDF4',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # adjust as needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
