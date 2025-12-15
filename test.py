import google.generativeai as genai

genai.configure(api_key="AIzaSyA_dI2UMbGcyRBfbIAcOt_XEETeW9Uc8BA")

model = genai.GenerativeModel('gemini-2.5-flash')

try:
    response = model.generate_content("Xin chào!")
    print("✅ API key hoạt động!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Lỗi: {e}")