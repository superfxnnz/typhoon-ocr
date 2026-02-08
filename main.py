import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "scb10x/typhoon-ocr-7b"
IMAGE_PATH = "test.jpeg" 

def load_model_and_processor():
    """แยกการโหลดออกมาเพื่อประสิทธิภาพสูงสุด"""
    print(f"--- 🚀 กำลังโหลด Model (ขั้นตอนนี้ทำครั้งเดียว) ---")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" # เสถียรและเร็วสำหรับ Windows
    )
    
    # ปรับแต่ง Model สำหรับการ Inference
    model.eval() 
    return model, processor

def run_typhoon_ocr(model, processor, image_path):
    if not os.path.exists(image_path):
        print(f"❌ ไม่พบไฟล์รูปภาพ: {image_path}")
        return

    # 1. เตรียมรูปภาพแบบรวดเร็ว
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this image accurately."},
            ],
        }
    ]

    # --- ส่วนเร่งความเร็ว (Visual Token Optimization) ---
    # บีบพิกเซลลงเล็กน้อยเพื่อให้ Process ไวขึ้น แต่ยังคงความชัดของ OCR
    # 600,000 พิกเซล คือจุดที่สมดุลที่สุดระหว่าง Speed และ Accuracy
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
        min_pixels=256 * 28 * 28,
        max_pixels=800 * 28 * 28  # ปรับเป็น 800 เพื่อความเร็วที่คงที่
    ).to("cuda")

    print(f"--- 🔍 เริ่มการแกะตัวอักษร (Tokens: {inputs.input_ids.shape[1]}) ---")
    
    start_time = time.time()
    
    # 2. รันการประมวลผล (จูนค่าสำหรับ Speed)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,             # สำคัญมาก: ช่วยให้เร็วขึ้นทวีคูณ
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # 3. ถอดรหัสผลลัพธ์
    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    result = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    elapsed_time = time.time() - start_time
    
    print("\n" + "="*40)
    print(f"⏱️ เวลาที่ใช้: {elapsed_time:.2f} วินาที")
    print("-" * 40)
    print(f"✨ ผลลัพธ์:\n{result}")
    print("="*40)
    
    # เคลียร์ Cache เพื่อคืน RAM ให้ระบบ (ถ้าต้องการรันต่อเนื่อง)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if torch.cuda.is_available():
        # โหลดโมเดลไว้ก่อน
        model, processor = load_model_and_processor()
        
        # รัน OCR
        run_typhoon_ocr(model, processor, IMAGE_PATH)
    else:
        print("❌ Error: ไม่พบ GPU")