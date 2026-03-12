"""Test image generation models."""
from dotenv import load_dotenv
load_dotenv(".env")
import os
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

models_to_try = [
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.5-flash-image", 
    "gemini-3-flash-preview",
]

for model_name in models_to_try:
    print(f"Trying {model_name}...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Generate an image of an adorable orange tabby cat wearing a tiny chef hat in a kitchen. Vertical portrait.",
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                data = part.inline_data
                print(f"  SUCCESS: {data.mime_type}, {len(data.data)} bytes")
                safe_name = model_name.replace("/", "_").replace(".", "_")
                with open(f"output/viral_cat/test_{safe_name}.png", "wb") as f:
                    f.write(data.data)
                print(f"  Saved!")
                break
            elif hasattr(part, "text") and part.text:
                print(f"  Text: {part.text[:150]}")
    except Exception as e:
        print(f"  FAILED: {str(e)[:200]}")

# Also try Imagen 4 via generate_images
print("\nTrying imagen-4.0-fast-generate-001 via generate_images...")
try:
    response = client.models.generate_images(
        model="imagen-4.0-fast-generate-001",
        prompt="An adorable orange tabby cat wearing a tiny chef hat in a kitchen, vertical portrait",
        config=genai.types.GenerateImagesConfig(
            number_of_images=1,
        ),
    )
    img = response.generated_images[0].image.image_bytes
    print(f"  SUCCESS: {len(img)} bytes")
    with open("output/viral_cat/test_imagen4.png", "wb") as f:
        f.write(img)
    print("  Saved!")
except Exception as e:
    print(f"  FAILED: {str(e)[:200]}")
