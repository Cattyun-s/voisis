import rclpy
from rclpy.node import Node
import threading
from std_msgs.msg import String
import websockets
import json
import asyncio
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import os


class Tablet_Bridge(Node):

    def __init__(self):
        super().__init__('tablet_bridge')
        self.websocket_clients = set()
        self.tts_ready_sub = self.create_subscription(
            String,
            "tts_ready",
            self.tts_callback,
            10
        )

        http_thread = threading.Thread(
            target=self.start_http_server,
            daemon=True
        )

        websocket_thread = threading.Thread(
            target = self.start_websocket_server,
            daemon=True
        )

        self.get_logger().info("Tablet Bridge started")

        http_thread.start()
        websocket_thread.start()

    def tts_callback(self,msg):
        filename = msg.data

        # self.get_logger().info(f"Nueva respuesta: {filename}")
        message = {"type":"tts",
                   "url":f"http://192.168.1.155:8000/{filename}"}
        self.send_to_tablet(message)

    def send_to_tablet(self,message):

        if len(self.websocket_clients) == 0:
            self.get_logger().warn(
                "No tablets connected"
            )
            return


        data=json.dumps(message)


        async def send_all():

            for client in self.websocket_clients:
                await client.send(data)


        asyncio.run_coroutine_threadsafe(
            send_all(),
            self.websocket_loop
        )

    def start_websocket_server(self):
        async def run_server():
        
            self.websocket_loop = asyncio.get_running_loop()
            async with websockets.serve(
                self.websocket_handler,
                "0.0.0.0",
                8765
            ):
                self.get_logger().info(
                    "WebSocket server running on port 8765"
                )

                await asyncio.Future()


        asyncio.run(run_server())

    def start_http_server(self):

        os.chdir(
            "/home/voice_ws/src/voisis/audio"
        )

        server = TCPServer(
            ("0.0.0.0",8000),
            SimpleHTTPRequestHandler
        )

        self.get_logger().info(
            "HTTP server running on port 8000"
        )

        server.serve_forever()

    async def websocket_handler(self, websocket):

        self.websocket_clients.add(websocket)

        self.get_logger().info(
            "Tablet CONNECTED"
        )

        try:

            while True:

                msg = await websocket.recv()

                self.get_logger().info(
                    f"Received: {msg}"
                )


        except websockets.exceptions.ConnectionClosed as e:

            self.get_logger().warn(
                f"Closed: {e.code} {e.reason}"
            )


        finally:

            self.websocket_clients.remove(websocket)

            self.get_logger().info(
                "Tablet DISCONNECTED"
            )

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = Tablet_Bridge()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()