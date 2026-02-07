from PIL import Image

# 1. เปิดไฟล์ภาพ
img = Image.open('your_image.jpeg')

# 2. คำนวณขนาดใหม่ (หาร 2)
# ใช้ // เพื่อให้ผลลัพธ์เป็นจำนวนเต็ม (Integer)
new_width = img.width // 2
new_height = img.height // 2

# 3. สั่ง Resize
# Image.LANCZOS ช่วยให้ภาพยังคงความคมชัดหลังลดขนาด
resized_img = img.resize((new_width, new_height), Image.LANCZOS)

# 4. บันทึกภาพ
resized_img.save('test.jpeg', quality=95)

print(f"ขนาดเดิม: {img.size} -> ขนาดใหม่: {resized_img.size}")