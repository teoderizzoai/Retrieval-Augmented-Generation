from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def get_generator(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", load_in_4bit=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_answer(pipeline, question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = pipeline(prompt, max_new_tokens=256, do_sample=True)
    return result[0]['generated_text']