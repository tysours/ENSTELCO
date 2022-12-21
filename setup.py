from enstelco import __version__
from setuptools import setup
from pathlib import Path

version = __version__

HERE = Path(__file__).parent

README = (HERE / 'README.md').read_text()

install_requires = (HERE / 'requirements.txt').read_text().splitlines()

python_requires = '>=3.6'

packages = ['enstelco', 'enstelco/elate']
package_data = {}

setup(name='enstelco',
      version=version,
      description='ENergy-STrain ELastic COnstant calculations made simple!',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://github.com/tysours/ENSTELCO',
      maintainer='Ty Sours',
      maintainer_email='tsours@ucdavis.edu',
      license='MIT',
      packages=packages,
      python_requires=python_requires,
      install_requires=install_requires,
      package_data=package_data,
      entry_points={'console_scripts': ['enstelco=enstelco.cli:main']},
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Chemistry'],
      project_urls={
          'Source': 'https://github.com/tysours/ENSTELCO',
          'Tracker': 'https://github.com/tysours/ENSTELCO/issues',
          },
      )
