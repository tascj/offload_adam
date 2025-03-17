from setuptools import setup, find_packages

setup(
    name="offload_adam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton>=3.1.0"
    ],
    author="tascj",
    author_email="tascj0@gmail.com",
    description="A Python package for offloaded Adam optimizer implementation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tascj/offload_adam",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 