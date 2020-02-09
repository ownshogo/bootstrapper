import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bootstrapper-SHOGO-OSAWA",
    version="0.1.0",
    author="Shogo Osawa",
    author_email="57659371+ownshogo@users.noreply.github.com",
    description="A Python library for bootstrap.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ownshogo/bootstrapper",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['joblib>=0.14.0', 'numpy>=1.17.4']
)
