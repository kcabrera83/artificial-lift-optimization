from setuptools import setup, find_packages

setup(
    name="artificial-lift-optimization",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
    ],
    author="Ing. Kelvin Cabrera",
    description="ML-based artificial lift optimization for oil wells",
    python_requires=">=3.9",
)
