from setuptools import setup
from setuptools import find_packages

# ofiginally based on 'keras-rl' by Matthias Plappert (https://github.com/matthiasplappert/keras-rl)

setup(name='unsupervised-drl',
      version='0.4.0',
      description='Boosting Reinforcement Learning with Unsupervised Feature Extraction',
      author='Simon Hakenes',
      author_email='simon.hakenes@ini.rub.de',
      url='https://gitlab.ruhr-uni-bochum.de/hakens4l/unsupervised-drl',
      license='MIT',
      install_requires=['keras>=2.0.7', 'gym[atari]', 'pillow', 'opencv-python', 'matplotlib'],
      packages=find_packages())
