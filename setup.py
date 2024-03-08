from setuptools import setup, find_packages

setup(
    name="yololab",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["assets/**"]},
    url="https://github.com/mojiexiaozi/yololab.git",
    license="Apache v2",
    author="yilin",
    author_email="ylin3759@gmail.com",
    description="yolov8 lab",
)
