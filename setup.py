# Setup Python module
from setuptools import setup, find_packages

modules = ["TALLSorts." + p for p in sorted(find_packages("./TALLSorts"))]

setup(
    name="TALLSorts",
    version="0.0.1",
    description="T-ALL Subtype Classifier/Investigator.",
    url="https://github.com/breons/TALLSorts",
    author="Allen Gu, Breon Schmidt",
    license="MIT",
    packages=["TALLSorts", *modules],
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "joblib==1.2.0",
        "matplotlib==3.6.2",
        "scikit-learn==1.1.3",
        "umap-learn==0.5.3",
        "plotly==5.11.0",
        "kaleido==0.2.1"
    ],
    entry_points={
          "console_scripts": ["TALLSorts=TALLSorts.tallsorts:run"]
    }
)
