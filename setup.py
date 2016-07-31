from setuptools import setup

setup(
    name='uvvid',
    version='0.0.1',
    packages=['uvvid'],
    install_requires=[
        'click',
        'numpy'
    ],
    entry_points='''
        [console_scripts]
        uvvid=uvvid.cli:cli
    ''',
)
