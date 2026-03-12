"""Test Nano Banana / Imagen API availability."""
from dotenv import load_dotenv
load_dotenv(".env")
import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# List image-capable models
print("=== Available image/gemini-2 models ===")
for model in client.models.list():
    name = model.name
    if "imagen" in name.lower() or "gemini-2" in name.lower() or "flash" in name.lower():
        methods = getattr(model, "supported_generation_methods", [])
        print(f"  {name} -> {methods}")

# Try Nano Banana via Gemini generate_content with image output
print("\n=== Testing Gemini image generation ===")
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents="Generate an image of an adorable orange tabby cat wearing a tiny chef hat, standing in a kitchen. Vertical portrait orientation.",
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            data = part.inline_data
            print(f"Got image: {data.mime_type}, {len(data.data)} bytes")
            with open("output/viral_cat/test_imagen.png", "wb") as f:
                f.write(data.data)
            print("Saved to output/viral_cat/test_imagen.png")
        elif hasattr(part, "text") and part.text:
            print(f"Text: {part.text[:200]}")
except Exception as e:
    print(f"Gemini 2.0 flash exp failed: {e}")

# Try gemini-2.0-flash
print("\n=== Testing gemini-2.0-flash ===")
try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Generate an image of an adorable orange tabby cat wearing a tiny chef hat, standing in a kitchen. Vertical portrait orientation.",
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            data = part.inline_data
            print(f"Got image: {data.mime_type}, {len(data.data)} bytes")
            with open("output/viral_cat/test_imagen2.png", "wb") as f:
                f.write(data.data)
            print("Saved to output/viral_cat/test_imagen2.png")
        elif hasattr(part, "text") and part.text:
            print(f"Text: {part.text[:200]}")
except Exception as e:
    print(f"Gemini 2.0 flash failed: {e}")
