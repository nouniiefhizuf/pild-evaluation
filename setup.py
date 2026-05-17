from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pild-evaluation",
    version="1.0.0",
    author="Ahmed Soltani, Ryan Chanchah, Skander Darghouth, Khalil Ben Rejeb",
    author_email="ahmed.soltani@medtech.tn",
    description="Evaluating LLMs as Actionable Physics Simulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/medtech-tn/pild-evaluation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=8.0", "black>=24.0", "flake8>=7.0", "mypy>=1.8"],
        "docs": ["mkdocs>=1.5", "mkdocs-material>=9.5"],
    },
    entry_points={
        "console_scripts": [
            "pild-eval=evaluation.cli:main",
        ],
    },
)
