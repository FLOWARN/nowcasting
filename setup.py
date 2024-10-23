from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='servir',
      version='1.0',
      description="Package containing code for Nowcasting in SERVIR project",
      long_description=long_description,
      author="",
      author_email="",
      packages=['servir'],
      package_dir={'servir': '/'},
      python_requires='>=3.6')