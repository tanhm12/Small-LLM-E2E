## Step 1: Build ggml 
Check README in ggml repo: 
```bash
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j4 gpt-neox
```

## Step 2: Convert and quantize model
1. Convert from huggingface model to ggml  
Inside the `ggml/build` folder, run:
```bash
python3 ../examples/gpt-neox/convert-h5-to-ggml.py ../../../train/models/pythia1b4-chat-oasst-dolly 1
```
2. Quantize model from 16 bit to 6 bit (q5_0)
```bash
./bin/gpt-neox-quantize ../../../train/models/pythia1b4-chat-oasst-dolly/ggml-model-f16.bin ../../../train/models/pythia1b4-chat-oasst-dolly/ggml-model-q5_0.bin q5_0
```
3. Verify if model is working
```bash
./bin/gpt-neox -m ../../../train/models/pythia1b4-chat-oasst-dolly/ggml-model-q5_0.bin -t 8 -n 256 -p "Human:\nWhat is your name?"
```

New ggml models are in the same folder with python model, you might want to move them to deploy folder to prepare for deployment step.