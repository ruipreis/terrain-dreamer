import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    installation_requirements = fr.readlines()

setuptools.setup(
    name="terrdreamer",
    version="0.0.0",
    author="Rui Reis.",
    author_email="rui.reis@dotmoovs.com",
    description="3D Terrain Generator",
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
