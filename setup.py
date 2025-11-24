from setuptools import setup, find_packages

setup(
    name="THESISQUADRUOED",
    version="0.1.0",
    description="PhD Thesis: Meta-RL for Fault Tolerance",
    author="Berkay Gürkan",
    packages=find_packages(),  # src altındaki __init__.py olan her şeyi bulur
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "gymnasium",
        "stable-baselines3",  # Faz 3.1 için SB3 kullanacağız
        "pyyaml",
        "pandas"
    ]
)