from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Sinhala text classification framework'
LONGER_DESCRIPTION = 'Makes text classification super easy !!'

lib_requirements = []

with open('./requirements.txt') as requirements_file:
    requirements_list = requirements_file.read().strip().splitlines()
    for requirement in requirements_list:
        # ignore commented-out lines
        if requirement.startswith('#'):
            continue
        elif requirement.startswith('-e '):
            lib_requirements.append(requirement.split('=')[1])
        else:
            lib_requirements.append(requirement)

# Setting up the library
setup(
        name="AdaptText",
        version=VERSION,
        author="Yathindra Kodithuwakku",
        author_email="yathindrarawya123@gmail.com",
        url="https://gitlab.com/loretex/loretexlib",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=lib_requirements,
        setup_requires=['wheel', 'pytest-runner'],
        tests_require=['pytest==4.4.1'],
        test_suite='tests',
        classifiers= [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Research",
            "Programming Language :: Python :: 3",
            "Operating System :: Unix",
        ]
)