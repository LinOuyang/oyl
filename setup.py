from setuptools import setup
setup(name='oyl',version='2.4.4',author='Lin Ouyang', 
    packages=["oyl","oyl.nn"], 
    include_package_data=True,
    install_requires=["numpy","matplotlib","cartopy","pandas",
                      "xarray","pyshp","sklearn"]
)
