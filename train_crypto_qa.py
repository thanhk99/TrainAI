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
def load_dataset_from_json(filename="crypto_qa_data.json"):
    """Tải dataset từ file JSON"""
    if not os.path.exists(filename):
        print(f"File {filename} không tồn tại.")
        return None
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = Dataset.from_dict(data)
    return dataset.train_test_split(test_size=0.2, seed=42)

# 2. Định dạng dữ liệu
def format_qa_examples(examples):
    """Định dạng dữ liệu hỏi-đáp"""
    texts = []
    for i in range(len(examples["question"])):
        text = f"Hỏi: {examples['question'][i]}\nĐáp: {examples['answer'][i]}\n"
        texts.append(text)
    return {"text": texts}

# 3. Tải dataset
print("Đang tải dataset...")
dataset = load_dataset_from_json()

if dataset is None:
    # Tạo dữ liệu mẫu nếu không có file
    print("Tạo dữ liệu mẫu...")
    sample_data = {
        "question": [
            "Bitcoin là gì?",
            "Sàn Binance là CEX hay DEX?",
            "Lợi ích của DEX so với CEX là gì?",
            "Ví tiền điện tử là gì?",
            "Staking là gì?"
        ],
        "answer": [
            "Bitcoin là đồng tiền điện tử đầu tiên và lớn nhất thế giới.",
            "Binance chủ yếu là một sàn giao dịch tập trung (CEX).",
            "DEX cho phép giao dịch không cần tin tưởng bên thứ ba.",
            "Ví tiền điện tử là nơi lưu trữ khóa cá nhân.",
            "Staking là quá trình giữ và khóa tiền điện tử."
        ]
    }
    dataset = Dataset.from_dict(sample_data)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")

# 4. Định dạng dữ liệu
formatted_train = dataset['train'].map(format_qa_examples, batched=True)
formatted_test = dataset['test'].map(format_qa_examples, batched=True)

# 5. Sử dụng model nhỏ, tương thích
model_name = "microsoft/DialoGPT-small"  # Model nhỏ, tương thích tốt
print(f"Đang tải model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 6. Tokenize
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        padding=True,
        truncation=True, 
        max_length=128,
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Đang tokenize dữ liệu...")
tokenized_train = formatted_train.map(tokenize_function, batched=True, remove_columns=formatted_train.column_names)
tokenized_test = formatted_test.map(tokenize_function, batched=True, remove_columns=formatted_test.column_names)

# 7. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 8. Tải model
model = AutoModelForCausalLM.from_pretrained(model_name)

# 9. Thiết lập training arguments cho phiên bản CŨ
# Sử dụng các tham số cơ bản, không dùng các tham số mới
training_args = TrainingArguments(
    output_dir="./crypto-qa-model",
    overwrite_output_dir=True,
    num_train_epochs=2,  # Giảm số epoch để training nhanh
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=50,
    no_cuda=not torch.cuda.is_available(),  # Sử dụng no_cuda thay vì use_cpu
    dataloader_num_workers=0,
    remove_unused_columns=False,
    
    # CÁC THAM SỐ MỚI - BỎ HOÀN TOÀN để tránh lỗi
    # evaluation_strategy="no",  # BỎ
    # save_strategy="steps",     # BỎ
    # load_best_model_at_end=False,  # BỎ
    # metric_for_best_model=None,    # BỎ
    # greater_is_better=False,       # BỎ
    # fp16=False,                    # BỎ
)

# 10. Khởi tạo Trainer (chỉ với train dataset)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 11. Training
print("Bắt đầu training...")
try:
    trainer.train()
    print("Training hoàn thành!")
    
    # Lưu model
    trainer.save_model()
    tokenizer.save_pretrained("./crypto-qa-model")
    print("Model được lưu tại ./crypto-qa-model")
    
except Exception as e:
    print(f"Lỗi trong quá trình training: {e}")
    print("Thử lưu model tại checkpoint...")
    try:
        trainer.save_model("./crypto-qa-model-checkpoint")
        tokenizer.save_pretrained("./crypto-qa-model-checkpoint")
        print("Model được lưu tại checkpoint")
    except:
        print("Không thể lưu model")

# 12. Test model đơn giản
print("\nTesting model...")
try:
    model.eval()
    test_question = "Bitcoin là gì?"
    
    # Tạo prompt đơn giản
    input_text = f"Hỏi: {test_question}\nĐáp:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
    
except Exception as e:
    print(f"Lỗi khi test model: {e}")

# Hiển thị thông tin phiên bản
print("\n" + "="*50)
print("THÔNG TIN HỆ THỐNG:")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except:
    print("Không thể xác định phiên bản transformers")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
except:
    print("Không thể xác định phiên bản torch")

print("="*50)