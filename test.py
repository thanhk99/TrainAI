import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def test_trained_model():
    print("🤖 TEST MODEL ĐÃ TRAIN")
    print("="*50)
    
    model_path = "./vietnamese-product-model"
    
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model đã train!")
        return
    
    # Tải model đã train
    print("🔄 Đang tải model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("✅ Đã tải model thành công!")
    
    # Câu hỏi test
    test_questions = [
        "Bơ lạt Anchor giá bao nhiêu?",
        "Sản phẩm Bơ lạt Anchor thuộc danh mục nào?",
        "Hạn sử dụng của Bơ lạt Anchor là gì?",
        "Bơ lạt Anchor có xuất xứ từ đâu?",
        "Mô tả về Bơ lạt Anchor",
        "Sản phẩm này còn bao nhiêu trong kho?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Câu hỏi: {question}")
        
        # Tạo prompt
        input_text = f"Hỏi: {question}\nĐáp:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode kết quả
        full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Tách phần trả lời
        if "Đáp:" in full_answer:
            answer_only = full_answer.split("Đáp:")[-1].strip()
            if "Hỏi:" in answer_only:
                answer_only = answer_only.split("Hỏi:")[0].strip()
        else:
            answer_only = full_answer
        
        print(f"✅ Trả lời: {answer_only}")
        print("-" * 80)

def interactive_mode():
    """Chế độ hỏi đáp tương tác"""
    print("\n" + "="*50)
    print("CHẾ ĐỘ HỎI ĐÁP TƯƠNG TÁC")
    print("Gõ 'quit' để thoát")
    print("="*50)
    
    model_path = "./vietnamese-product-model"
    
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model đã train!")
        return
    
    # Tải model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    while True:
        question = input("\n🤔 Nhập câu hỏi: ").strip()
        
        if question.lower() in ['quit', 'exit', 'thoát', 'q']:
            print("👋 Tạm biệt!")
            break
            
        if not question:
            continue
            
        try:
            input_text = f"Hỏi: {question}\nĐáp:"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )
            
            full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Đáp:" in full_answer:
                answer_only = full_answer.split("Đáp:")[-1].strip()
                if "Hỏi:" in answer_only:
                    answer_only = answer_only.split("Hỏi:")[0].strip()
            else:
                answer_only = full_answer
            
            print(f"🤖 Trả lời: {answer_only}")
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    test_trained_model()
    interactive_mode()