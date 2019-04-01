import setuptools

setuptools.setup(name='gym_rl_book',
                 version='0.0.6',
                 install_requires=['gym', 'numpy', 'tabulate', 'matplotlib'],
                 packages=setuptools.find_packages(),
                 long_description=open('README.md').read(),
                 url='https://github.com/rl-study/gym-rl-book',
                 author='void-main',
                 author_email='voidmain1313113@gmail.com',
                 )
