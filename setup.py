from setuptools import setup

setup(
    name="particle",
    version="0.0.1",
    description="Simulates a particle model",
    url="",
    author="Me",
    author_email="test@g",
    license="MIT",
    packages=["particle"],
    install_requires=[
        "coolname",
        "itertools",
        "matplotlib",
        "numba",
        "numpy",
        "pre-commit",
        "pyarrow",
        "seaborn",
        "scipy",
        "yaml",
    ],
    zip_safe=False,
)
