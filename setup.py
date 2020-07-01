import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prsm", # Replace with your own username
    version="0.0.1",
    author="Benjamin Landon",
    author_email="bc.landon@gmail.com",
    description="A package for PCA using random matrix theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/landon23/prsm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)