import os
import csv
import json
import torch
import requests
import transformers
import pyreft
from huggingface_hub import login


def send_control_request(instruction):
    url = f"http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    prompt = prompt_no_input_template % instruction

    data = {
        "model": "llama3",  # Base model
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get('response', '')
        return results
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    
# Function to load the Hugging Face API key from a file

def load_hf_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("API key file not found.")
        return None
    
# Path to your API key file
api_key_file_path = 'huggingfacekey.txt'

# Load the API key
hf_api_key = load_hf_api_key(api_key_file_path)

if hf_api_key:
    # Use the API key to log in
    login(hf_api_key)
    print("Logged in to Hugging Face Hub successfully.")
else:
    print("Failed to load Hugging Face API key.")

def load_training_examples(filename='questions_answers.csv'):
    training_examples = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                # Check if the answer field is not empty
                if len(row) >= 2 and row[1].strip():
                    training_examples.append([row[0], row[1]])  # Assuming row format is correct
                else:
                    print(f"Skipped row with empty answer: {row}")
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return training_examples


########FINE TUNING FUNCTION START############

training_examples = load_training_examples()

##### Fine-tune the model
output_csv_file = 'generated_answers.csv'

# Define the prompt template
prompt_no_input_template = """<s>[INST] <<SYS>>
You are a computer networking expert specialized in Cisco ACI ( Application Specific Infrastructure ).
<</SYS>>

%s [/INST]
"""

# Load the model
model_name_or_path = "meta-llama/Llama-3.2-3B"
device = "mps"
#device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)

# Set pad_token as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Set up the ReFT config
reft_config = pyreft.ReftConfig(representations={
    "layer": 15, "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=4)})
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device(device)
reft_model.print_trainable_parameters()

# Create data module
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, [prompt_no_input_template % e[0] for e in training_examples], 
    [e[1] for e in training_examples])

# Define training arguments
training_args = transformers.TrainingArguments(
    num_train_epochs=100.0, output_dir="./tmp", per_device_train_batch_size=10, 
    learning_rate=4e-3, logging_steps=20)

# Train the model
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
_ = trainer.train()

def load_questions_from_csv(filename='output.csv'):
    questions = []
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            row_count = 0  # Debug: count the rows processed
            for row in reader:
                questions.append(row[0])  # Assuming the question is in the first column
                row_count += 1
            print(f"Total questions loaded: {row_count}")  # Debug: print total questions loaded
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return questions

# Load questions for inference testing
questions = load_questions_from_csv()

# Open CSV file for writing the results
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Fine-tuned Answer', 'Base Model Answer'])

    for question in questions:
        instruction = question
        # Tokenize and prepare the input
        prompt = prompt_no_input_template % instruction
        prompt = tokenizer(prompt, return_tensors="pt").to(device)

        base_unit_location = prompt["input_ids"].shape[-1] - 1  # Last position
        _, reft_response = reft_model.generate(
            prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
            intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
            eos_token_id=tokenizer.eos_token_id, early_stopping=True
        )
        
        fine_tuned_answer = tokenizer.decode(reft_response[0], skip_special_tokens=True)
        
        base_model_response = send_control_request(instruction)
        
        writer.writerow([question, fine_tuned_answer, base_model_response])
        print(f"Question: {question}")
        print(f"Fine-tuned Answer: {fine_tuned_answer}")
        print(f"Base Model Answer: {base_model_response}\n")

print("Interactive mode. Type your questions or 'exit' to quit.")

while True:
    instruction = input("You: ")
    
    if instruction.lower() == "exit":
        print("Goodbye!")
        break

    # Tokenize and prepare the input
    prompt = prompt_no_input_template % instruction
    prompt = tokenizer(prompt, return_tensors="pt").to(device)
    
    base_unit_location = prompt["input_ids"].shape[-1] - 1  # Last position
    _, reft_response = reft_model.generate(
        prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
        eos_token_id=tokenizer.eos_token_id, early_stopping=True
    )
    
    fine_tuned_answer = tokenizer.decode(reft_response[0], skip_special_tokens=True)
    
    base_model_response = send_control_request(instruction)
    
    print(f"Question: {instruction}")
    print(f"Fine-tuned Answer: {fine_tuned_answer}")
    print(f"Base Model Answer: {base_model_response}\n")

# Save and publish model
reft_model.set_device("cpu")  # Send back to CPU before saving.
reft_model.save(
    save_directory="./CiscoDevNetSandboxRunningConfig", 
    save_to_hf_hub=True, 
    hf_repo_name="veeransooran/aci_model_test_1"
)