from setuptools import setup

install_reqs = ['numpy', 'matplotlib']
dep_links = []

setup(
    name='bb_fb_stitcher',
    version='0.0.0.dev1',
    description='Stitch images from different cam positions,'
                'with an affine transformation',
    long_description='',
    entry_points={
        'console_scripts': [
            'bb_fg_subtract = fb_stitcher.scripts.bb_fg_subtract:main',
            'bb_fb_stitcher = fb_stitcher.scripts.bb_fb_stitcher:main',
            'bb_stitch_videos = fb_stitcher.scripts.bb_stitch_videos:main',
            'bb_stitch_images = fb_stitcher.scripts.bb_stitch_images:main'
        ]
    },
    url='https://github.com/gitmirgut/bb_affine_stitcher',
    author='gitmirgut',
    author_email="gitmirgut@users.noreply.github.com",
    packages=['fb_stitcher', 'fb_stitcher.scripts'],
    install_requires=install_reqs,
    dependency_links=dep_links,
    license='GNU GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5'
    ]
)
