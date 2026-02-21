from transformers import pipeline

print("=== 5. Локальная LLM (TinyLlama) ===")
# Загружаем маленькую модель для локального запуска
llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

prompt = "<|system|>\nТы полезный ИИ-ассистент. Отвечай кратко на русском языке.</s>\n<|user|>\nНазови 3 главных преимущества машинного обучения.</s>\n<|assistant|>\n"

result = llm(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9)
print(result[0]['generated_text'].split("<|assistant|>\n")[-1])