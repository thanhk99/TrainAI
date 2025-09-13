import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

def load_trained_model(model_path="./crypto-qa-model"):
    """Load model đã train"""
    print("Đang tải model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return tokenizer, model, device

def chat_with_model(tokenizer, model, device, question, max_length=150):
    """Chat với model"""
    # Tạo prompt theo format đã train
    input_text = f"Hỏi: {question}\nĐáp:"
    
    # Tokenize
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128
    ).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            repetition_penalty=1.1  # Giảm lặp lại
        )
    
    # Decode và trích xuất câu trả lời
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Chỉ lấy phần sau "Đáp:"
    if "Đáp:" in full_response:
        answer = full_response.split("Đáp:")[1].strip()
    else:
        answer = full_response.replace(input_text, "").strip()
    
    return answer

# 2. Code chính để chat
def main():
    # Load model đã train
    tokenizer, model, device = load_trained_model()
    print("Model đã được tải thành công!")
    print(f"Device: {device}")
    print("\n" + "="*50)
    print("CHATBOT CRYPTO QA - Gõ 'thoát' để dừng")
    print("="*50)
    
    while True:
        # Nhập câu hỏi
        question = input("\nBạn: ").strip()
        
        if question.lower() in ['thoát', 'exit', 'quit', 'q']:
            print("Kết thúc chat!")
            break
            
        if not question:
            continue
        
        # Generate answer
        try:
            answer = chat_with_model(tokenizer, model, device, question)
            print(f"Bot: {answer}")
            
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()