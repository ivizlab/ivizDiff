from setuptools import find_packages, setup

install_requires = [
    "diffusers>=0.35.0",
    "transformers>=4.56.0",
    "accelerate>=1.10.0",
    "omegaconf",
    "fire",
]

setup(
    name="ivizdiff",
    version="1.0.0",
    description="iVizDiff — real-time AI video transformation for live performance",
    license="Apache 2.0",
    package_dir={"": "."},
    packages=find_packages("."),
    python_requires=">=3.10.0",
    install_requires=install_requires,
)
