from setuptools import setup

setup(name='gym_rl_book',
      version='0.0.1',
      install_requires=['gym', 'numpy', 'tabulate'],
      packages=['gym_rl_book'],
      long_description=open('README.md').read(),
)
