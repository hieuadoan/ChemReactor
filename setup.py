from setuptools import setup, find_packages

setup(
    name="chemreactor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=7.0.0",
            "sphinx_rtd_theme>=1.3.0",
        ],
    },
    python_requires=">=3.12",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for chemical reactor simulation and visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chemreactor",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
