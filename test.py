from google import genai
import os

client = genai.Client(api_key="AIzaSyBT8Ax34W7c5-JiPoC7ZPQa_V7T3l9BQ9w")

def chat_with_gemini(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt
        )
        
        return response.text
    
    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}"

# Chạy thử
if __name__ == "__main__":
    user_input = "Giải thích thuyết tương đối hẹp trong 2 câu."
    print("Gemini đang trả lời...\n")
    print(chat_with_gemini(user_input))