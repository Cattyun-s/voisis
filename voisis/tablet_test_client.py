import asyncio
import websockets


async def main():

    async with websockets.connect(
        "ws://localhost:8765"
    ) as ws:

        print("CONNECTED")

        while True:
            await asyncio.sleep(1)


asyncio.run(main())