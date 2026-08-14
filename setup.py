from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'voisis'

setup(
 name=package_name,
 version='0.0.0',
 packages=find_packages(exclude=['test']),
 data_files=[
     ('share/ament_index/resource_index/packages',
             ['resource/' + package_name]),
     ('share/' + package_name, ['package.xml']),
     (os.path.join('share', package_name, 'launch'),
     glob('launch/*.py')),
   ],
 install_requires=['setuptools'],
 zip_safe=True,
 maintainer='TODO',
 maintainer_email='TODO',
 description='TODO: Package description',
 license='TODO: License declaration',
 tests_require=['pytest'],
 entry_points={
     'console_scripts': [
             'smartVAD_Node = voisis.smartVAD_Node:main',
             'whisper_Node = voisis.whisper_Node:main',
             'BERT_Node = voisis.BERT_Node:main',
             'Piper_Node = voisis.Piper_Node:main',
             'tablet_bridge = voisis.tablet_bridge:main'
     ],
   },
)