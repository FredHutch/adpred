import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name = "adpred",
    version="1.2.8",
    author="Ariel Erijman",
    author_email="aerijman@fredhutch.org, aerijman@neb.com",
    description="Prediction of Transcription Activation Domains from protein sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FredHutch/adpred-pkg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'keras==2.8.0',
        'numpy==1.22.2',
        'pandas==1.4.1',
        'plotly==5.6.0',
        'requests==2.27.1',
        'scikit-learn==1.0.2',
        'tensorflow==2.9.3',
    ],
    include_package_data=True,
    scripts=[
        'bin/run-adpred'
    ]
)
