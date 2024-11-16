import PyPDF2
import requests
import csv
import json
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

# Function to generate questions and answers using the LLAMA model
def generate_qa(text, model_url, num_questions=10):
    chunk_size = 5000  # Adjust based on your model's token limit
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    

    for chunk in text_chunks:
        questions_answers = []
        # Formulate the prompt
        prompt = (
            f"Read the following text and generate {num_questions} pairs of questions and answers. "
            f"Generate randomized questions about the data to form a dataset being used to fine-tune a model"
            f"Make your answers verbose to include all the details about the questions you generrate"      
            f"Only include questions that can be answered based on the given data. Do not include questions without answers.\n"
            f"Make sure your questions are like this:\n"
            f"Q: What is the purpose of Reverse Filter ports when configuring filters in Cisco ACI?\n"
            f"A: Reverse Filter ports should be used always when Apply Both Directions is enabled, and they deploy exactly as defined for both consumer-to-provider and provider-to-consumer directions with source and destination ports reversed\n"
            
            f"Ensure the questions are clear and the answers are explicitly found in the text. Text:\n\n{chunk}"
            f"Format each question with a 'Q: ' prefix and each answer with an 'A: ' prefix on the next line. Only include questions that can be answered based on the given data. Do not include questions without answers.\n"
        )
        data = {
            "model" : "llama3.2:3b-instruct-fp16",
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0
        }
        # Send a request to the LLAMA model
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(model_url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get('response', '')
        lines = results.strip().split("\n")
        formatted_pairs = []
        question = None
        for line in lines:
            if line.startswith("Q") and ":" in line:
                question = line.split(":", 1)[1].strip()  # Extract question
            elif line.startswith("A") and question:
                answer = line.split(":", 1)[1].strip()  # Extract answer
                formatted_pairs.append([question, answer])  # Append as list of [question, answer]
                question = None  # Reset question for the next pair
        save_to_csv(formatted_pairs, "questions_answers.csv")
        
        
        
        
    return True

# Function to save questions and answers to a CSV file
def save_to_csv(questions_answers, filename="output.csv", mode='a'):
    existing_pairs = set()
    
    if os.path.exists(filename):
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header row if it exists
            for row in reader:
                existing_pairs.add(tuple(row))

    with open(filename, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        # Write the header if the file is empty
        if os.stat(filename).st_size == 0:
            writer.writerow(["Question", "Fine-tuned Answer"])
            print("Header written to CSV file.")
        
        # Write each question-answer pair, avoiding duplicates
        for item in questions_answers:
            if len(item) == 2:
                question, answer = item
                if (question, answer) not in existing_pairs:
                    print(f"Writing to CSV: {question} | {answer}")  # Debugging output
                    writer.writerow([question, answer])
                    existing_pairs.add((question, answer))

# Main script
if __name__ == "__main__":
    # File paths
    folder_path = "docs/"
    pdf_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.pdf')]
    
    output_csv_path = "questions_answers.csv"
    model_url = "http://localhost:11434/api/generate"  # Replace with your Ollama model's local URL

    # Extract text from PDF
    for pdf in pdf_files:
        pdf_text = extract_text_from_pdf(f"{folder_path}{pdf}")
        qa_pairs = generate_qa(pdf_text, model_url)

    # Generate questions and answers
    

    
    

    print(f"Questions and answers saved to {output_csv_path}")
