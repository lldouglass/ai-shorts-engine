"""Test Imagen 4 Ultra - Google's best image model."""
from dotenv import load_dotenv
load_dotenv(".env")
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

models = [
    ("imagen-4.0-ultra-generate-001", "Imagen 4 Ultra ($0.06/img)"),
    ("imagen-4.0-generate-001", "Imagen 4 Standard ($0.04/img)"),
    ("imagen-4.0-fast-generate-001", "Imagen 4 Fast ($0.02/img)"),
]

prompt = "An adorable orange tabby cat with bright green eyes wearing a tiny white chef hat, standing on a kitchen counter next to a mixing bowl, warm golden lighting, flour dusted on its nose, photorealistic, vertical portrait 9:16 aspect ratio"

for model_id, label in models:
    print(f"\nTesting {label} ({model_id})...")
    try:
        response = client.models.generate_images(
            model=model_id,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
            ),
        )
        img = response.generated_images[0].image.image_bytes
        safe_name = model_id.replace(".", "_").replace("-", "_")
        out_path = f"output/viral_cat/test_{safe_name}.png"
        with open(out_path, "wb") as f:
            f.write(img)
        print(f"  SUCCESS: {len(img):,} bytes -> {out_path}")
    except Exception as e:
        print(f"  FAILED: {str(e)[:250]}")
