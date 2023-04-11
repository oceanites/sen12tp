from setuptools import setup, find_packages

__author__ = "Thomas Roßberg"

with open("requirements.txt", "r") as file:
    requirements = file.readlines()

setup(
    name="sen12tp",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False,
    package_data={"": ["requirements.txt", "readme.md", "LICENSE"]},
)
