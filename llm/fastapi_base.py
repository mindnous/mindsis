# Required packages:
# pip install fastapi uvicorn pydantic transformers torch ctransformers
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
import time
import uuid
import json

import sys
import pathlib
FILEPATH = pathlib.Path(__file__).parent.absolute()
sys.path.append(f'{FILEPATH}/../model/')


def read_llm_info():
    with open(f'{FILEPATH}/llm_info.json') as rj:
        return json.load(rj)


class BaseConfig():
    pass


cfg = BaseConfig()

app = FastAPI(title="Local LLM API with OpenAI Compatibility")

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Option 1: Using Hugging Face transformers
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype="auto",
#     device_map="auto",
#     low_cpu_mem_usage=True
# )
model = None

# Define Pydantic models to match OpenAI API format
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

def generate_response(prompt, max_tokens=1024, temperature=0.7, top_p=1.0):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs.input_ids.shape[1]
    
    # Generate output
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    
    # Process the output
    response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
    completion_tokens = outputs.shape[1] - input_tokens
    
    return response, input_tokens, completion_tokens

def format_prompt(messages):
    """Format the messages into a prompt the model can understand."""
    formatted_prompt = ""
    
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<|system|>\n{message.content}\n"
        elif message.role == "user":
            formatted_prompt += f"<|user|>\n{message.content}\n"
        elif message.role == "assistant":
            formatted_prompt += f"<|assistant|>\n{message.content}\n"
    
    # Add the final assistant marker to indicate where the model should generate
    formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Format the prompt from the messages
        prompt = format_prompt(request.messages)
        
        # Generate the response
        response_text, prompt_tokens, completion_tokens = generate_response(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Create the response object
        return ChatCompletionResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    llm_info = read_llm_info()
    """List available models endpoint to maintain compatibility."""
    return {
        "object": "list",
        "data": [
            {
                "id": llm_info['modelname'],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization-owner"
            }
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    llm_info = read_llm_info()
    return {"status": "healthy", "model": llm_info['modelname']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)