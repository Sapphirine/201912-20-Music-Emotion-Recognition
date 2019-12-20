from setuptools import setup, find_packages

setup(name='mer',
      version='0.1',
      description='Music Emotion Recognition',
      url='http://github.com/j-cahill/mer',
      author='Jesse Cahill, Zile Wang, Zhongling Jiang',
      author_email='jcahill225@gmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6, <3.8',
      install_requires=[
            'numpy',
            'scikit-learn',
            'pandas',
            'tqdm',
            'librosa',
            'future'
      ],
      zip_safe=False)