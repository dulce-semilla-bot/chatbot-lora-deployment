from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__)

# Cargar el modelo base desde Hugging Face
model_name = "NousResearch/Llama-2-7b-chat-hf"
lora_model_name = "14OVER/mi-modelo-chatbot"  # Tu repositorio de Hugging Face

# Cargar el modelo base
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", offload_folder="offload")

# Cargar los adaptadores LoRA desde Hugging Face
model = PeftModel.from_pretrained(model, lora_model_name)

# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Crear el pipeline de generación de texto
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100, batch_size=1)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data['prompt']
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    result = pipe(formatted_prompt)
    response = result[0]['generated_text'].replace(formatted_prompt, "").strip()
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
