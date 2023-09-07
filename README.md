# Gaussian Viewer

This repository contains a proof-of-concept web viewer for [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), a technique capable of rasterizing realistic scenes in real-time.

It combines the original paper [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) implementation with hardware H264 encoding and WebRTC streaming to stream interactive views in realtime.

## Live Demo

A live viewer is available at:
- [Hugging Face](https://huggingface.co/spaces/dylanebert/gaussian-viewer)

## Prerequisites

- Python 3 or above
- NVIDIA card with display driver 525.xx.xx or above
- CUDA Toolkit 11.7

and least one of the following:
- Ubuntu 22.04 (Linux)
- WSL2 with Ubuntu 22.04 (Windows)
- Docker

## Cloning this Repository

To clone this repository and its dependencies, check it out with

```
git clone https://github.com/dylanebert/gaussian-viewer.git --recursive
```

Be sure to use the `--recursive` flag to clone the local dependencies:
- [gaussian-viewer-frontend](https://github.com/dylanebert/gaussian-viewer-frontend)
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [VideoProcessingFramework](https://github.com/NVIDIA/VideoProcessingFramework)

## Downloading Models

This project is currently hardcoded to run the bicycle example from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset.

Click [here](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) to download the 12GB Mip-NeRF 360 dataset. Unzip the archive, then copy the contained `bicycle` folder to this repository's `models` folder.

## Running locally

This project depends on the [VideoProcessingFramework](https://github.com/NVIDIA/VideoProcessingFramework) for hardware tensor-to-video encoding and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) for rasterization. VideoProcessingFramework in particular has very specific requirements, including Ubuntu 22.04, or WSL2 with Ubuntu 22.04 on Windows. Alternatively, you can follow the Docker instructions to run in other environments.

### Without Docker

1. Create a virtual python environment

```
python -m venv venv
```

2. Install the required python packages

```
python -m pip install -r requirements.txt
python -m pip install diff-gaussian-rasterization/
python -m pip install VideoProcessingFramework/
python -m pip install VideoProcessingFramework/src/PytorchNvCodec/
```

If you encounter issues, be sure the required submodules are present

```
git submodule update --init --recursive
```

and refer to corresponding submodule documentation for troubleshooting.

3. Run the server

```
python -m uvicorn main:app
```

This should print a local web address running the demo (i.e. `http://localhost:8000`)

### With Docker

1. Edit the Dockerfile

The provided Dockerfile is configured to serve the viewer with an SSL certificate over HTTPS. This isn't necessary for running locally.

Replace the last two lines

```
EXPOSE 443

CMD ["python3.10", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "privkey.pem", "--ssl-certfile", "fullchain.pem"]
```

with

```
EXPOSE 8000

CMD ["python3.10", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Build the container

```
docker build -t gaussian-viewer .
```

3. Run the container

```
docker run --gpus all -p 8000:8000 gaussian-viewer
```

### STUN/TURN

WebRTC typically requires STUN/TURN servers to function on web. However, this shouldn't be an issue when running locally.

The provided implementation is configured to fetch ICE servers from [twilio](https://www.twilio.com/en-us) using private credentials. You can provide your own credentials, replace them with other ICE servers, or ignore them when running locally.

## Project Layout

This project consists of a python backend and svelte frontend.

- Backend
  - main.py
    - src/
      - camera.py 
      - gaussian\_model.py
      - render.py
      - turn.py
      - utils.py
- Frontend (gaussian-viewer-frontend/)
 - src/
   - src/routes/+page.svelte
 - public/

The backend server entrypoint is `main.py`.

All functional frontend code is in `+page.svelte`. This is built to static HTML that is served in `public/`.

## Community

If you're interested in community discussion on this project, join the [Hugging Face discord](https://hf.co/join/discord) and feel free to ping me @IndividualKex. Alternatively, email me at [individualkex@gmail.com](mailto:individualkex@gmail.com).