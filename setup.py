from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='KRNN_SF',
      version='0.1',
      description='KRNN with spatial features',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Beta',
              'License :: GPL3',
              'Programming Language :: Python :: 3.6',
              'Topic :: Image Processing'],
      url='http://github.com/gykovacs/krnn_with_spatial_features',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      license='GPL3',
      packages=['KRNN_SF'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy'
              ],
      test_suite='nose.collector',
      tests_require=['nose'],
      py_modules=['KRNN_SF'],
      zip_safe=False)
