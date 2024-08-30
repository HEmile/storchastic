import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="storchastic",  # Replace with your own username
    version="0.3.7",
    author="Emile van Krieken",
    author_email="emilevankrieken.com",
    description="Stochastic Deep Learning for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HEmile/storchastic",
    install_requires=["entmax", "pyro-ppl", "torch", "packaging"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
