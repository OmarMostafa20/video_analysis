from setuptools import setup, find_packages

setup(
    name='face-emotion-detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'keras',
        'mtcnn',
        'opencv-python'
    ],
    entry_points={
        'console_scripts': [
            'detect-emotions=face_emotion_detection.detect:main',
        ],
    },
    author='Omar Mostafa',
    author_email='omar.mostafa@robustastudio.com',
    description='A package for detecting faces and predicting emotions from video.',
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
