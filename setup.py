import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='base_query_and_analytics',
    version='0.1.9',
    author='Anil Gurbuz',
    author_email='anil.gurbuz@newcrest.com.au',
    description='Utilities for querying and analytics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://newcrestmining.visualstudio.com/Cadia%20PCA/_git/base_query_and_analytics',
    packages=['base_query_and_analytics'],
    package_data={'':['html_templates/*.html', 'html_templates/*.png','styles/*.jpg','styles/*.css']},
    include_package_data=True,
    install_requires=[
      'joblib ==1.1.1',
      'scipy ==1.10.0',
      'plotly ==5.14.1',
      'piconnect ==0.9.1',
      'colorama ==0.4.4',
      'tqdm ==4.64.0'
    ],
    conda_deps=[
  'appdirs ==1.4.4',
  'blas',
  'bottleneck ==1.3.4',
  'brotlipy ==0.7.0',
  'bzip2 ==1.0.8',
  'ca-certificates ==2023.01.10',
  'certifi ==2022.12.7',
  'cffi ==1.15.1',
  'charset-normalizer ==2.0.4',
  'cryptography ==38.0.4',
  'fftw ==3.3.9',
  'future ==0.18.2',
  'icc_rt ==2022.1.0',
  'idna ==3.4',
  'intel-openmp ==2021.4.0',
  'jinja2 ==3.1.2',
  'libffi ==3.4.2',
  'libzlib ==1.2.12',
  'markupsafe ==2.1.1',
  'mkl ==2021.4.0',
  'mkl-service ==2.4.0',
  'mkl_fft ==1.3.1',
  'mkl_random ==1.2.2',
  'numexpr ==2.8.1',
  'numpy ==1.21.5',
  'numpy-base ==1.21.5',
  'openssl ==1.1.1s',
  'packaging ==21.3',
  'pandas ==1.4.1',
  'pip ==21.2.4',
  'pooch ==1.4.0',
  'pycparser ==2.21',
  'pyopenssl ==22.0.0',
  'pyparsing ==3.0.4',
  'pysocks ==1.7.1',
  'python ==3.9.12',
  'python-dateutil ==2.8.2',
  'python_abi ==3.9',
  'pythonnet ==2.5.2',
  'pytz ==2021.3',
  'pyyaml ==6.0',
  'requests ==2.28.1',
  'setuptools ==61.2.0',
  'six ==1.16.0',
  'sqlite ==3.38.2',
  'tbb ==2021.5.0',
  'tenacity ==8.2.2',
  'tk ==8.6.12',
  'tzdata ==2022a',
  'urllib3 ==1.26.14',
  'vc ==14.2',
  'vs2015_runtime ==14.27.29016',
  'wheel ==0.37.1',
  'win_inet_pton ==1.1.0',
  'wincertstore ==0.2',
  'wrapt ==1.14.0',
  'xz ==5.2.6',
  'yaml ==0.2.5',
  'colorama ===0.4.4',
  'tqdm ===4.64.0',
],
    conda_channel=[
      'plotly',
      'conda - forge',
      'anaconda',
      'defaults']
)
