from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

from ctransformers import AutoModelForCausalLM
import os 
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

NUM_THREADS = int(os.environ.get("NUM_THREADS", 4))
LLM_LIB = os.environ.get("LLM_LIB", "avx")
LOAD_ON_FLY = bool(os.environ.get("LOAD_ON_FLY", False))


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b-deduped')

class ChatInput(BaseModel):
    text: str
    stream: bool = False
    
    top_k: int = 40
    top_p: float = 0.95
    temperature: float = 0.5
    repetition_penalty: float = 1.1
    seed: int = -1

prompt_format = "Human:\n{human}" + tokenizer.eos_token + "\nAssistant:\n"

if not LOAD_ON_FLY:
    llm  = AutoModelForCausalLM.from_pretrained('pythia1b4-chat-oasst-dolly/ggml-model-q5_0.bin', model_type='gpt_neox', local_files_only=True, lib=LLM_LIB)
    

def chat(chat_inp: ChatInput):
    if not LOAD_ON_FLY:
        global llm
    else:
        llm = AutoModelForCausalLM.from_pretrained('pythia1b4-chat-oasst-dolly/ggml-model-q5_0.bin', model_type='gpt_neox', local_files_only=True, lib=LLM_LIB)
        
    text = chat_inp.text
    text = prompt_format.format(human=text)
    tokens = tokenizer.encode(text)
    
    def stream_resp():
        for token in llm.generate(tokens, threads=NUM_THREADS, **chat_inp.dict(exclude={"text", "stream"})):
            yield tokenizer.decode(token)
        
    if not chat_inp.stream:
        # res = llm.generate(tokens, threads=NUM_THREADS, **chat.dict(exclude={"text", "stream"}))
        # return "".join(tokenizer.batch_decode(res, skip_special_tokens=True))
        return stream_resp()
    else:
        return StreamingResponse(stream_resp(), media_type='text/event-stream')

@app.get("/chat")
async def chat_get(chat_inp: ChatInput=  Depends()):
    return chat(chat_inp)


@app.post("/chat")
async def chat_post(chat_inp: ChatInput):
    return chat(chat_inp)
    

from _logging import setup_logging
@app.on_event("startup")
def startup_event():
    setup_logging("log/app.log")
    
    
# from uvicorn.workers import UvicornWorker

# class SingleUvicornWorker(UvicornWorker):
#     CONFIG_KWARGS = UvicornWorker.CONFIG_KWARGS
#     CONFIG_KWARGS["limit_concurrency"] =  2
