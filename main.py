# 20-35 วินาที แต่ตัวหนังสือไม่ถูก

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import time
import os

MODEL_ID = "scb10x/typhoon-ocr-7b"
IMAGE_PATH = "test.jpeg"

def load_model_and_processor():
    print(f"--- 🚀 Load Model (Optimized for 8GB VRAM) ---")
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # ลองเปลี่ยนเป็น float16 เพื่อลด overhead บนโน้ตบุ๊ก
    bnb_4bit_use_double_quant=True # กลับมาเปิดใช้เพื่อประหยัด VRAM เพิ่มอีกประมาณ 200-400MB
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", # เปลี่ยนจาก "cuda" เป็น "auto" เพื่อให้จัดการ memory ได้ฉลาดขึ้น
    trust_remote_code=True,
    torch_dtype=torch.float16, # ใช้ float16 ให้แมตช์กับ bnb_config
    attn_implementation="sdpa"
    )
    model.eval()
    return model, processor

def run_typhoon_ocr(model, processor, image_path):
    if not os.path.exists(image_path): return
    
    # จัดการรูปภาพ
    raw_image = Image.open(image_path).convert("RGB")
    
    # 🎯 จุดแก้ 1: ลด Pixel Limit
    # 336 * 28 * 28 คือจุดที่คุ้มค่าที่สุด (Sweet Spot) สำหรับโมเดล 7B 
    # ลดจำนวน Token ของภาพลงได้มหาศาล แต่ยังอ่านภาษาไทยชัด
    pixel_limit = 768 * 28 * 28 

    messages = [{"role": "user", "content": [
        {"type": "image", "image": raw_image},
        {"type": "text", "text": "Extract all Thai text accurately."} 
    ]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # เตรียม Inputs
    inputs = processor(
        text=[text],
        images=[raw_image],
        return_tensors="pt",
        min_pixels=256 * 28 * 28,
        max_pixels=pixel_limit
    ).to("cuda")

    print(f"--- ⚙️ Processing Image... ---")
    start_time = time.time()
    
    # 🎯 จุดแก้ 2: Generate Settings
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2412, # 🎯 ลดจาก 1024 ถ้าข้อความในภาพไม่ได้ยาวเป็นหน้ากระดาษ
            do_sample=False,    # ปิดการสุ่มเพื่อความไวและแม่นยำ
            use_cache=True,     # สำคัญมาก ต้องเปิดไว้เสมอ
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    elapsed = time.time() - start_time
    print("-" * 30)
    print(f"✨ ผลลัพธ์:\n{result}")
    print(f"\n⏱️ เวลาที่ใช้: {elapsed:.2f} วินาที")

if __name__ == "__main__":
    # 🎯 จุดแก้ 3: ลบ torch.compile ออกก่อน
    # บน Windows torch.compile มักจะทำให้การรันครั้งแรกช้าไป 5-10 นาที (สะสมในเวลาที่เห็น)
    model, processor = load_model_and_processor()
    
    run_typhoon_ocr(model, processor, IMAGE_PATH)