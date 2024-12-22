from setuptools import setup, find_packages

setup(
    name="heatchmap",
    version="0.1.0",
    author="Hitchwiki",
    author_email="info@hitchwiki.org",
    description="A package for hitchhiking data analysis and visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hitchwiki/hitchmap-data",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "folium",
        "geopandas",
        "ipykernel",
        "ipython",
        "matplotlib",
        "numpy",
        "osmnx",
        "pandas",
        "scikit-learn",
        "scipy",
        "shapely",
        "tqdm",
        "lingua-language-detector",
    ],
)
