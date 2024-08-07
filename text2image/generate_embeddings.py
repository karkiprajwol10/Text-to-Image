from transformers import AutoModel, AutoTokenizer 
import torch
import translators as ts
import translators.server as tss

# Define the model repo
model_name = "prajjwal1/bert-mini" 


# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_embed(text):
    text = tss.google(text, to_language="en")
    # Transform input tokens 
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    # Model apply
    last_hidden_state = model_output.last_hidden_state

    first_token_embeddings = last_hidden_state[:,0,:]

    return first_token_embeddings
