# coding=utf-8
from setuptools import setup, find_packages
from distutils.core import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
fsdir = path.abspath(path.join(here, '..', 'src'))
vendordir = path.abspath(path.join(here, '..', 'vendor'))
fs_builddir = path.abspath(path.join(here, '..', '.build'))


# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def main():
    setup(
        name='basilisk',
        version='0.1.0',
        description='basilisk: The concept-based potential heuristic learner from Basel',
        long_description=long_description,
        url='https://github.com/aibasel/concept-based-heuristics',
        author='Augusto B. Corrêa and Cedric Geissmann and Florian Pommerening and Guillem Francès',
        author_email='-',

        keywords='planning logic STRIPS heuristic',
        classifiers=[
            'Development Status :: 3 - Alpha',

            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',

            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],

        packages=find_packages('src'),  # include all packages under src
        package_dir={'': 'src'},  # tell distutils packages are under src

        install_requires=[
            'setuptools',
            "tarski @ git+https://git@github.com/aig-upf/tarski.git@2eda780#egg=tarski-devel",
            "sltp @ git+https://git@github.com/aig-upf/sltp.git@6b7f2ab#egg=sltp-for-basilisk",
            'keras',
            'numpy',
            'sklearn',
            'tensorflow',
            'matplotlib',
        ],


        extras_require={
            'dev': ['pytest', 'tox'],
            'test': ['pytest', 'tox'],
        },
    )


if __name__ == '__main__':
    main()
