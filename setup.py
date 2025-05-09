from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='servir',
      version='1.0',
      description="Package containing code for Nowcasting in SERVIR project",
      long_description=long_description,
      author="",
      author_email="",
      packages=['servir', 'servir_data_utils', 'servir_nowcasting_examples'],
      package_dir={'servir': '/',
                   'servir_data_utils': '/',
                   'servir_nowcasting_examples':'/'},
      python_requires='>=3.8')