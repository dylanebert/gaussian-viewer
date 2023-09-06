import json
import logging
import time
import traceback

import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.WARN)

from src import Camera, GaussianModel, Renderer, get_ice_servers

model_path = "models/bicycle/point_cloud/iteration_30000/point_cloud.ply"
camera_path = "models/bicycle/cameras.json"

sessions = {}

# load gaussian model
gaussian_model = GaussianModel().load(model_path)


# load camera info
with open(camera_path, "r") as f:
    cam_info = json.load(f)[12]


# initialize server
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_frame(container, data):
    try:
        packets = container.parse(data)
        for packet in packets:
            frames = container.decode(packet)
            for frame in frames:
                return frame
    except Exception as e:
        logging.error(e)
        traceback.print_exc()

    return None


class Offer(BaseModel):
    sdp: str
    type: str


class Session:
    session_id: str
    renderer: Renderer

    def __init__(self, session_id: str, renderer: Renderer):
        self.session_id = session_id
        self.renderer = renderer


class FrameProducer(VideoStreamTrack):
    kind = "video"

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

        container = av.CodecContext.create("h264", "r")
        container.pix_fmt = "yuv420p"
        container.width = session.renderer.camera.image_width
        container.height = session.renderer.camera.image_height
        container.bit_rate = 14000000
        container.options = {"preset": "ultrafast", "tune": "zerolatency"}

        self.container = container

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        failed_attempts = 0
        max_failed_attempts = 10

        while True:
            try:
                start_time = time.time()
                data = self.session.renderer.render()
                logging.info(f"Render time: {time.time() - start_time}")
                if data is not None and len(data) > 0:
                    frame = parse_frame(self.container, data)
                    if frame is not None:
                        break
                    else:
                        raise Exception("Error parsing frame")
            except Exception as e:
                logging.error(e)
                logging.debug(traceback.format_exc())
                failed_attempts += 1
                if failed_attempts >= max_failed_attempts:
                    logging.error(f"Failed to render frame after {failed_attempts} attempts")
                    break

        frame.pts = pts
        frame.time_base = time_base

        return frame


@app.post("/offer")
async def create_offer(offer: Offer, session_id: str = Query(...)):
    logging.info(f"Creating offer for session {session_id}")

    session = sessions.get(session_id, None)
    if session is None:
        camera = Camera().load(cam_info)
        renderer = Renderer(gaussian_model, camera, logging=False)
        session = Session(session_id, renderer)
        sessions[session_id] = session

    pc = RTCPeerConnection()
    track = FrameProducer(session)
    pc.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            del sessions[session_id]

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            payload = json.loads(message)
            logging.info(f"Received payload: {payload}")

            if payload["type"] == "camera_update":
                position = payload["position"]
                rotation = payload["rotation"]
                rotation = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
                track.session.renderer.update(position, rotation)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

    answer = await pc.createAnswer()

    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/ice-servers")
async def get_ice():
    return get_ice_servers()
