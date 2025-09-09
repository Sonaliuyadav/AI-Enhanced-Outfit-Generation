from setuptools import setup
import os

APP = ['main.py']  # your Flask entry point
DATA_FILES = ['updated_recommendation.csv']  # include your dataset
OPTIONS = {
    'argv_emulation': True,
    'packages': ['torch', 'scikit-learn', 'flask', 'transformers', 'diffusers', 'tqdm', 'pandas', 'Pillow'],
    'includes': ['torch', 'scikit-learn', 'flask', 'transformers', 'diffusers', 'tqdm', 'pandas', 'Pillow'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
