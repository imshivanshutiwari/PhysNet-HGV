import os
from setuptools import setup, find_packages

setup(
    name="physnet-hgv",
    version="0.1.0",
    description="Physics-Informed Neural Kalman Framework for Real-Time Tracking of Maneuvering Hypersonic Glide Vehicles Under Plasma Blackout Conditions",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip()
    ],
    python_requires=">=3.11",
)
