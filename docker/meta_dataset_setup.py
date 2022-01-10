import setuptools

setuptools.setup(
    name='meta_dataset',
    version='0.1',
    packages=setuptools.find_packages(),
    install_requires=[
	'absl-py>=0.7',
	'gin-config>=0.1.2',
	'numpy>=1.13.3',
	'scipy>=1.0.0',
	'six >= 1.10',
	# 'tensorflow>=1.12.0',
    ]
)
