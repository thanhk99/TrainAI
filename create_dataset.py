from datasets import load_dataset
import json

# Your Hugging Face API key
api_key = "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"

# Load the dataset from Hugging Face
dataset = load_dataset("mkly/crypto-sales-question-answers", token=api_key)

# Convert the dataset to a JSON-serializable format
data = dataset['train'].to_dict()

# Save the data to a JSON file
with open('crypto_qa_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Dataset has been downloaded and saved as 'crypto_sales_question_answers.json'")