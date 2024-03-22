# Following https://roboticsbackend.com/ros-import-python-module-from-another-package/
# and https://wiki.ros.org/rospy_tutorials/Tutorials/Makefile

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['cap'],
    package_dir={'': 'src'},
)

setup(**setup_args)