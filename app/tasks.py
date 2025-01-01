import os
#from huggingface_hub import HfFolder

# Set the environment variable for Hugging Face cache
#os.environ["HF_HOME"] = "D:\\huggingface_cache"  # Use any desired directory

# Alternatively, set Hugging Face's default directory
#HfFolder.save_token(os.environ["HF_HOME"])
from app.model import SpamCheckModel
from transformers import GPT2Config, GPT2Model
import tiktoken
import torch
import torch.nn as nn

config = GPT2Config()
m1 = GPT2Model(config)


class Gptforclassification(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.gpt = model
        self.linear = nn.Linear(768, num_classes)

        for pram in self.gpt.parameters():
            pram.requires_grad = False

        for pram in self.gpt.h[-1].parameters():
            pram.requires_grad = True

        for pram in self.linear.parameters():
            pram.requires_grad = True

    def forward(self, x):
        x = self.gpt(x).last_hidden_state
        x = x[:, -1, :]  # Get last token representation
        x = self.linear(x)
        return x


def load_model(weights_path, num_classes=2):
    base_model = GPT2Model.from_pretrained("gpt2",cache_dir="D:\\huggingface_cache")
    model = Gptforclassification(base_model, num_classes)
    model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    model.eval()
    return model

weights_path = "D:\downloads\model_weights.pth"
tokenizer = tiktoken.get_encoding("gpt2")
checker = load_model(weights_path)

def store_checkreq(t):
    model = SpamCheckModel(text = t.text)
    model.save()
    return model.id
'''
def run_check(t_id: id):
    model = SpamCheckModel.get_by_id(t_id)
    p = model.text
    input_ids = tokenizer.encode(p)
    input_ids = torch.tensor(input_ids)
    with torch.no_grad():
        logits = model(input_data)  # Output of the classification head
        predictions = torch.argmax(logits, dim=1)  # Predicted class index
    if prediction == 0:
        reply = "Not Spam"
    else:
        reply = "Spam"  
    
    model.output = reply
    model.save()  
'''
def run_check(t_id: int):
    try:
        print(f"Starting run_check for ID: {t_id}")

        # Retrieve the model entry from the database
        spam_check_model = SpamCheckModel.get_by_id(t_id)
        if not spam_check_model:
            print(f"No model found for ID: {t_id}")
            return

        p = spam_check_model.text
        print(f"Retrieved text: {p}")

        # Tokenize the input text
        input_ids = torch.tensor([tokenizer.encode(p)])  # Add batch dimension
        print(f"Tokenized input IDs: {input_ids}")

        # Make predictions
        with torch.no_grad():
            logits = checker(input_ids)  # Output from classification head
            print(f"Logits: {logits}")

            predictions = torch.argmax(logits, dim=1)  # Predicted class index
            print(f"Predictions: {predictions}")

        # Determine the reply based on predictions
        if predictions.item() == 0:
            reply = "Not Spam"
        else:
            reply = "Spam"
        
        print(f"Predicted reply: {reply}")

        # Save the result to the database
        spam_check_model.output = reply
        spam_check_model.save()
        print(f"Reply saved for ID: {t_id}")

    except Exception as e:
        print(f"Error in run_check: {e}")

def find(t_id: int):
    model = SpamCheckModel.get_by_id(t_id)

    reply = model.output
    if reply is None:
        reply = "Processing...."
    return reply    