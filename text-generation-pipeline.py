from transformers import pipeline
from model_evaluation import compute_perplexity, evaluate

generator = pipeline("text-generation", model="distilgpt2")

prompt = "what is the population of United States?"
system_prompt = "Answer the following question clearly and concisely."

full_input = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

print("Generating...")

response = generator(full_input, max_new_tokens=150)[0]['generated_text'].split("Assistant:")[-1].strip()
print(response)

with open("generated-text.txt", "w") as file:
        file.write(response)

print("Done Generating")
print("---------------------")
print("Starting Evaluation")

with open("generated-text.txt", "r") as file:
    generated_text = file.read()
    evaluation = evaluate(prompt, generated_text)
print(evaluation)

exit(0)