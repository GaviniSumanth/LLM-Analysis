from ollama import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
from rouge import Rouge
import os
import json

load_dotenv()

text_data = open('./data/news.txt','r').read()

def download_models(models):
    for model in models:
        client = Client(host=os.environ["OLLAMA_BASE_URL"])
        if model not in [i["model"] for i in client.list()["models"]]:
            print(f"Downloading model: {model}")
            client.pull(model=model)
        else:
            print(f"{model} is already downloaded.")


def get_score(summary, reference_summary):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(str(summary), str(reference_summary))
    return rouge_scores[0]


def get_summary(model_name):
    model = OllamaLLM(model=model_name, base_url=os.environ["OLLAMA_BASE_URL"])
    template = """
        Use the following pieces of context to generate a summary highliting the most important takeaways.

        {text}

        Summary:
    """
    prompt = PromptTemplate.from_template(template)
    rag_chain = {"text": RunnablePassthrough()} | prompt | model | StrOutputParser()
    summary = rag_chain.invoke(text_data)
    return summary


def evaluate_models(models):
    scores = []
    for model in models:
        print(f"Evaluating model: {model}")
        summary = get_summary(model)
        score = get_score(text_data, summary)
        scores.append({"model": model, "summary": summary, "rouge_score": score})
    return scores


models = [
    "gemma2:2b",
    "llama3.2:1b",
    "phi3:3.8b",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "tinyllama:1.1b",
    "llama3.2:3b",
]

print("### PREPARING MODELS...")
download_models(models)

print("### EVALUATING MODELS...")
evaluation = evaluate_models(models)

output = json.dumps(evaluation, sort_keys=True, indent=4)
with open("./results/text_summarization.json", "w") as out_file:
    out_file.write(output)
