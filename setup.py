from setuptools import setup, find_packages

setup(
    name="gaze_estimate",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "scipy",
        "tqdm",
    ],
    python_requires=">=3.6",
) 