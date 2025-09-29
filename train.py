import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Tải dataset từ file JSON
def load_product_dataset_from_json(filename="data.json"):
    """Tải dataset sản phẩm từ file JSON"""
    if not os.path.exists(filename):
        print(f"File {filename} không tồn tại.")
        return None
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Chuyển đổi dữ liệu thành format phù hợp
    formatted_data = {
        "text": []
    }
    
    for product in data:
        # Tạo mô tả chi tiết từ các trường dữ liệu
        product_text = f"Sản phẩm: {product['name']}\n"
        product_text += f"Danh mục: {product['category']}\n"
        product_text += f"Giá: {product['price']:,.0f} VND\n"
        product_text += f"Số lượng: {product['quantity']} {product['unit']}\n"
        product_text += f"Mô tả: {product['description']}\n"
        product_text += f"Thông tin chi tiết: {product['content']}\n"
        product_text += f"Hạn sử dụng: {product['expery']}\n"
        product_text += "---\n"
        
        formatted_data["text"].append(product_text)
    
    dataset = Dataset.from_dict(formatted_data)
    return dataset.train_test_split(test_size=0.2, seed=42)

# 2. Tải dataset
print("Đang tải dataset sản phẩm...")
dataset = load_product_dataset_from_json("data.json")

if dataset is None:
    raise Exception("Không tìm thấy file data.json. Vui lòng kiểm tra đường dẫn.")

print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")

# 3. SỬ DỤNG MODEL HỖ TRỢ TIẾNG VIỆT
# Lựa chọn 1: Model của VinAI (tốt nhất cho tiếng Việt)
# model_name = "VietAI/gpt-j-6b-vietnamese-news" 
# Lựa chọn 2: Model đa ngôn ngữ
model_name = "bigscience/bloom-560m"  # Model nhỏ hơn, hỗ trợ đa ngôn ngữ
# Lựa chọn 3: Model XLM-Roberta cho classification
# model_name = "xlm-roberta-base"

print(f"Đang tải model: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Thêm token đặc biệt cho tiếng Việt
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Đảm bảo tokenizer hỗ trợ tiếng Việt
    tokenizer.add_special_tokens({'additional_special_tokens': ['<vi>', '</vi>']})
    
except Exception as e:
    print(f"Lỗi khi tải tokenizer: {e}")
    # Fallback to multilingual tokenizer
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# 4. Tokenize với cấu hình cho tiếng Việt
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        padding="max_length",
        truncation=True, 
        max_length=512,  # Tăng max_length cho tiếng Việt
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Đang tokenize dữ liệu sản phẩm...")
tokenized_train = dataset['train'].map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)
tokenized_test = dataset['test'].map(tokenize_function, batched=True, remove_columns=dataset['test'].column_names)

# 5. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 6. Tải model
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Resize token embeddings nếu cần
    model.resize_token_embeddings(len(tokenizer))
    
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    # Fallback to smaller model
    model_name = "microsoft/DialoGPT-small"
    model = AutoModelForCausalLM.from_pretrained(model_name)

# 7. Thiết lập training arguments cho tiếng Việt
training_args = TrainingArguments(
    output_dir="./vietnamese-product-model",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Tăng epoch cho tiếng Việt
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=3e-5,  # Learning rate thấp hơn cho fine-tuning
    weight_decay=0.01,
    warmup_steps=100,
    no_cuda=not torch.cuda.is_available(),
    dataloader_num_workers=0,
    remove_unused_columns=False,
    gradient_accumulation_steps=2,  # Tích luỹ gradient cho batch lớn hơn
)

# 8. Khởi tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 9. Training
print("Bắt đầu training model tiếng Việt...")
try:
    train_result = trainer.train()
    print("Training hoàn thành!")
    
    # Lưu model
    trainer.save_model()
    tokenizer.save_pretrained("./vietnamese-product-model")
    print("Model được lưu tại ./vietnamese-product-model")
    
except Exception as e:
    print(f"Lỗi trong quá trình training: {e}")