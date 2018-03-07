from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='regseg',
      version='0.1',
      description='KRNN with spatial features',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: GPL3',
              'Programming Language :: Python :: 3.6',
              'Topic :: Image Processing'],
      url='http://github.com/gykovacs/krnn_with_spatial_features',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      license='GPL3',
      packages=['krnn_with_spatial_features'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy'
              ],
      test_suite='nose.collector',
      tests_require=['nose'],
      py_modules=['krnn_with_spatial_features'],
      zip_safe=False)
