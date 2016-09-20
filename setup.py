from setuptools import setup, find_packages

install_reqs = []
dep_links = []

setup(
    name='bb_affine_stitcher',
    version='0.0.0.dev1',
    description='Stitch images from different cam positions, with an afffine transformation',
    long_description='',
    # entry_points={
    #     'console_scripts': [
    #         'bb_composer = composer.scripts.bb_composer:main'
    #     ]
    # },
    url='https://github.com/gitmirgut/bb_affine_stitcher',
    author='gitmirgut',
    author_email="gitmirgut@users.noreply.github.com",
    packages=['affine_stitcher'],
    install_requires=install_reqs,
    dependency_links=dep_links,
    license='GNU GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5'
    ]
)
