from setuptools import setup, find_packages

setup(
    name='audio-analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pyannote.audio',
        'torchaudio',
        'pydub',
        'whisper',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'audio-to-chunks=audio_analysis.audio_to_chunks:main',
            'audio-to-text=audio_analysis.audio_to_text:main',
        ],
    },
    author='Omar Mostafa',
    author_email='omar.mostafa@robustastudio.com',
    description='A package for audio chunking and transcription.',
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
