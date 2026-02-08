import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import time
import os

MODEL_ID = "scb10x/typhoon-ocr-7b"
IMAGE_PATH = "test.jpeg"

def load_model_and_processor():
    print(f"--- 🚀 Loading Model with Optimization ---")
    
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
        device_map="cuda", # บังคับลง GPU เท่านั้น (ถ้า error แปลว่า VRAM ไม่พอ)
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # ถ้าใช้ RTX 3000/4000 ให้เปลี่ยนเป็น "flash_attention_2"
        attn_implementation="sdpa" 
    )
    
    model.eval()
    return model, processor

def run_typhoon_ocr(model, processor, image_path):
    if not os.path.exists(image_path): return

    image = Image.open(image_path).convert("RGB")
    
    # 1. ลดพิกเซลลง (448*28*28 ≈ 156,800 pixels) 
    # นี่คือจุดเปลี่ยนชีวิตเรื่องความเร็ว!
    min_pixels = 256 * 28 * 28
    max_pixels = 448 * 28 * 28 

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Extract text"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        min_pixels=min_pixels,
        max_pixels=max_pixels
    ).to("cuda")

    start_time = time.time()
    
    # 2. ปรับการ Generate ให้กระชับขึ้น
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512, # ลดจาก 1024 ถ้าข้อความในรูปไม่ได้เยอะมาก
            do_sample=False,
            use_cache=True,
        )

    # 3. Decode เฉพาะส่วนที่สร้างใหม่
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"⏱️ Time taken: {time.time() - start_time:.2f} seconds")
    print(f"✨ Result: {result}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        model, processor = load_model_and_processor()
        run_typhoon_ocr(model, processor, IMAGE_PATH)