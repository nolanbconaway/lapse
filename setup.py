import setuptools

setuptools.setup(
    name="lapsepy",
    version="0.0.1",
    py_modules=["lapse"],
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "imageio>=2.10.3",
        "opencv-python>=4.5.1.48",
        "numpy>=1.20.2",
        "tqdm>=4.60.0",
    ],
)