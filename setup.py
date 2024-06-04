from setuptools import find_packages, setup


# Collect packages
packages = find_packages(exclude=('tests', 'experiments'))
print(f'Found the following packages to be created:\n  {packages}')

# Set up the package
setup(
    name='rephrl',
    version='1.0.0',
    packages=packages,
    python_requires='>=3.9.0',
    url='https://github.com/ruyianry/rep_hierarchy_rl',
    author='Ruyi An',
)