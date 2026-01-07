from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from PIL import Image
from langchain_core.messages import HumanMessage
import base64
from io import BytesIO
load_dotenv()


llm = ChatOpenAI(
    model="nxchat-vision", 
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
    temperature=1
)

image = Image.open(r"C:\Users\ha.phan\Downloads\Screenshot 2026-01-07 144202.png")
buffered = BytesIO()
image.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()
message = HumanMessage(
    content=[
        {"type": "text", "text": PASS1_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
    ]
)
response = llm.invoke([message])
print(response.content)