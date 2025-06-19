from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util
import torch
import math

# Load model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_perplexity(text, model_name="gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Tokenize input
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    # Prevent memory overflow for long texts
    max_length = model.config.n_positions
    stride = 512
    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        trg_len = end_loc - i

        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100  # mask loss on prefix

        with torch.no_grad():
            outputs = model(input_ids_slice, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def compute_embedding_similarity(prompt, response):
    embeddings = embedding_model.encode([prompt, response], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity


def evaluate(prompt, output, coherence_threshold=50, relevance_threshold=0.7):
    coherence_score = compute_perplexity(output)
    relevance_score = compute_embedding_similarity(prompt, output)

    # Combine into one score (weighted)
    combined_score = (1 / (1 + coherence_score)) * 0.5 + relevance_score * 0.5

    result = {
        "coherence": coherence_score,
        "relevance": relevance_score,
        "combined_score": round(combined_score, 3),
        "coherent": coherence_score < coherence_threshold,
        "relevant": relevance_score > relevance_threshold,
        "needs_rerun": coherence_score > coherence_threshold or relevance_score < relevance_threshold
    }
    return result