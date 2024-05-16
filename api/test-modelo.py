from huggingface_hub import login
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch


def test_pregunta(pregunta, model, tokenizer):
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    )
    #prompt = "[INST]¿Podrías ayudarme a crear una expresión cron para ejecutar un script todos los lunes y los martes a las 8:00 AM?[/INST]"
    prompt = "[INST]"+pregunta+"[/INST]"
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



def cargar_modelo():
    # Load base model(Mistral 7B)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )
    #model = AutoModelForCausalLM.from_pretrained("messalhi/mistral_gestorjobs", quantization_config=bnb_config, device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained("messalhi/mistral_gestorjobs", quantization_config=bnb_config, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained("messalhi/mistral_gestorjobs")
    return model, tokenizer

def filtrar_respuesta(respuesta):
    #respuesta = "[INST]Dame una expresión cron para los martes y los miercoles a las 10:00[/INST] La expresión cron sería 0 10 1,4 * *.[/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] [/END] ["
    try:
        respuesta = respuesta.split("[/INST]")
        respuesta = respuesta[1]
        respuesta = respuesta.split("[/END]")
        respuesta = respuesta[0]
        return respuesta
    except:
        return respuesta

if __name__== "__main__":
    login("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")
    modelo, tokenizer = cargar_modelo()
    pregunta = "¿Podrías ayudarme a crear una expresión cron para ejecutar un script todos los lunes y los martes a las 8:00 AM?"
    print("-----")
    respuesta = test_pregunta(pregunta, modelo, tokenizer)
    print(filtrar_respuesta(respuesta))
