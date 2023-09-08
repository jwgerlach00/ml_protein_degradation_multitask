from setuptools import setup, find_packages


setup(
    name='linkerology_multitask',
    version='0.1',
    license='MIT',
    author='Jacob Gerlach',
    author_email='jwgerlach00@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # package_data={'naclo': [
    #     'assets/bleach_model.yml',
    #     'assets/bleach_warnings.yml',
    #     'assets/binarize_default_params.json',
    #     'assets/binarize_default_options.json',
    #     'assets/recognized_bleach_options.json',
    #     'assets/recognized_binarize_options.json',
    #     'assets/recognized_units.json',
    #     'assets/recognized_salts.json',
    # ]},
    python_requires='>=3.8.0',
    install_requires=[
    ],
)
