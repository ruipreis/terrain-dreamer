import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    installation_requirements = fr.readlines()

setuptools.setup(
    name="terrdreamer",
    version="0.0.0",
    author="Rui Reis, Diogo Monteiro, Filipe Fernandes",
    description="Terrdreamer is a Python package for fast, realistic 3D terrain generation using generative models, DEMs, and RGB satellite imagery. Ideal for researchers, game developers, and GIS professionals, it supports GPU acceleration. Unleash your creativity with terrdreamer!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruipreis/terrain-dreamer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=installation_requirements,
)
