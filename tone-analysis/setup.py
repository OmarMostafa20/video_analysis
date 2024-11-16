from setuptools import setup, find_packages

setup(
    name='tone-analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'keras',
        'librosa',
        'numpy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'tone-analysis=tone_analysis.tone_analysis:main',
        ],
    },
    author='Omar Mostafa',
    author_email='omar.mostafa@robustastudio.com',
    description='A package for emotion recognition from audio files.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://gitlab.com/robustastudio/ai/momrah-research/-/tree/video-analysis?ref_type=heads',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
