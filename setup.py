import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name = "adpred",
    version="1.3.2",
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
        'keras==3.11.3',
        'numpy==2.3.3',
        'pandas==2.3.2',
        'plotly==6.3.0',
        'requests==2.32.5',
        'requests-oauthlib==2.0.0',
        'scikit-learn==1.7.2',
        'tensorflow==2.20.0'
    ],
    include_package_data=True,
    scripts=[
        'bin/run-adpred'
    ]
)
