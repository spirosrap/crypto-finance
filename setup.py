from setuptools import setup, find_packages

setup(
    name="crypto-finance",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'ta-lib'
    ]
) 