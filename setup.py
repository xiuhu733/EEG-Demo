from setuptools import setup, find_packages

setup(
    name="eeg-deep-learning",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'mne',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorboard',
        'wandb',
        'tqdm',
        'pyyaml',
        'titans-pytorch'
    ]
) 