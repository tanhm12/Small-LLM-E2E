This folder contains all necessary scripts and files to deploy a ggml compiled model for a simple chat application.
## Features
- Efficient on CPU: (1.4b) quantized model allows to run on machine with limited resources and no GPU is required
- Low ram usage: 1.25GB per model load; if using LOAD_ON_FLY --> only 120MB is required, the rest will be loaded in inference time.
- Dockerized deployment: packed into a single Docker image, simplifying the deployment process.

## Getting Started
###  Prerequisites
1. Docker version 20.10.22 or above
2. (Optional) For running locally, you would need Poetry, see [Poetry home page](https://python-poetry.org/docs/) for running a python project with Poetry
### Installation
1. Move the built model in build step to `pythia1b4-chat-oasst-dolly/ggml-model-q5_0.bin` or download it at my huggingface space ([Zayt/pythia1b4-chat-oasst-dolly](https://huggingface.co/Zayt/pythia1b4-chat-oasst-dolly))
2. Build the docker image, see `Dockerfile` and `docker/build.sh` for more detail.
```bash
sh docker/build.sh
```
If the command above ran successfully, the image is ready (default name is `tanhm/pythia-engbot:latest`, check it with command `docker images`).  
TODO: upload prebuilt docker image
### Configuration
See .env.example for example .env file if you intend to run it locally.  
Else check the `docker/docker-compse.yml.example` for  example docker-compse.yml file. Copy and paste into new `docker/docker-compse.yml`. You might want to tweak some environment variables listed below:  
- NUM_THREADS: 4  # num threads to run model per worker
- LLM_LIB: avx   # change to "basic" if encounter core dumped 
- LOAD_ON_FLY: False  #  whether or not to load model into ram each time requesting, enable to save a RAM but inference speed will be slower a bit.
- NUM_WORKERS: 1  # number of workers (model replicas)
### Running
- Change working 
- See `run.sh` for running locally.
## Known issues
- ggml models are not thread-safe, so do not try to use single model instance to  inference parallelly --> Need to use rate limit tool 
- Basic rate limit doesn't seem to work, tried `blocked-async`, `gunicorn backlog`, `threading.Lock`
--> Need to use other rate limit tools: Redis, nginx, custome gateway server, ...