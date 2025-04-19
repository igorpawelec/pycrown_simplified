import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycrown_simplified",
    version="0.1",
    author="Igor Pawelec",
    author_email="igor.pawelec@student.urk.edu.pl",
    description="Simplified tree crown segmentation using CHM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igorpawelec/pycrown_simplified",
    license="GPLv3",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "scikit-image>=0.20",
        "rasterio>=1.3",
        "numba>=0.60",
        "fiona>=1.9",
        "networkx>=3.0",
    ],
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
