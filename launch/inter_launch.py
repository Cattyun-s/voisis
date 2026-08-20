from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    pkg = 'voice_ws'

    BERT_Node = Node(
        package = pkg,
        executable = 'BERT_Node',
        output = 'screen'
    )

    piper_node = Node(
        package = pkg,
        executable = 'Piper_Node',
        output = 'screen'
    )

    smart_node = Node(
        package = pkg,
        executable = 'smartVAD_Node',
        output = 'screen'
    )

    whisper_node = Node(
        package = pkg,
        executable = 'whisper_Node',
        output = 'screen'
    )

    return LaunchDescription([
        BERT_Node,
        piper_node,
        smart_node,
        whisper_node
    ])

def main(args=None):
   voise_assis = generate_launch_description()
   try:
       LaunchDescription(voise_assis)
# 
   except KeyboardInterrupt:
       print("\nStopping...")
       Shutdown(voise_assis)
# 
if __name__ == '__main__':
   main()