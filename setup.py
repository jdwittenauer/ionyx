from setuptools import setup
from setuptools import find_packages

setup(name='ionyx',
      version='0.0.1',
      description='High-level machine learning experimentation library.',
      author='John Wittenauer',
      author_email='jdwittenauer@gmail.com',
      url='https://github.com/jdwittenauer/ionyx',
      license='Apache',
      install_requires=['numpy', 'matplotlib', 'pandas', 'seaborn', 'scikit-learn'],
      packages=find_packages())
