from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

app = FastAPI()

class PreguntaRequest(BaseModel):
    pregunta: str

def test_pregunta(pregunta, model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    prompt = "[INST]" + pregunta + "[/INST]"
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    return sequences[0]['generated_text']

modelo = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global modelo, tokenizer
    modelo, tokenizer = cargar_modelo()

def cargar_modelo():
    # Load base model(Mistral 7B)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained("messalhi/mistral_gestorjobs", quantization_config=bnb_config, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained("messalhi/mistral_gestorjobs")
    return model, tokenizer

def filtrar_respuesta(respuesta):
    try:
        respuesta1 = respuesta.split("[/INST]")
        respuesta1 = respuesta1[1]
        respuesta1 = respuesta1.split("[/END]")
        respuesta1 = respuesta1[0]
        return respuesta1
    except:
        return respuesta

@app.post("/test/")
async def test_endpoint(pregunta_request: PreguntaRequest):
    global modelo, tokenizer
    if modelo is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado correctamente")
    respuesta = test_pregunta(pregunta_request.pregunta, modelo, tokenizer)
    print("Respuesta original" + respuesta)
    respuesta2 = filtrar_respuesta(respuesta)
    print("Respuesta limpiada " + respuesta2)
    return {"respuesta": respuesta2}
