import os
import json
from google import genai
from pydantic import BaseModel, Field
from PIL import Image

# Setup: Get your API Key from https://aistudio.google.com/
# Set it as an environment variable or a Replit Secret
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ Error: GEMINI_API_KEY not found. Please set it in your environment.")

# Initialize the new Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiUnavailableError(Exception):
    """Exception raised when the Gemini LLM service is unavailable (e.g. 503 High Demand)."""
    pass

# Define schemas for structured outputs
class ReceiptItem(BaseModel):
    name: str = Field(description="The name of the product.")
    amount_per_item: str | None = Field(None, description="The volume/weight (e.g., 500g, 1L) if specified.")
    quantity: str = Field("1", description="The number of items purchased.")
    price: str = Field(description="The unit price of the item.")
    total_price: str = Field(description="The total price for that line item.")

class ReceiptData(BaseModel):
    items: list[ReceiptItem] = Field(description="A list of items purchased.")
    total_amount_on_receipt: str = Field(description="The grand total amount to be paid.")

def extract_receipt_data(image_source):
    """
    Uses Gemini 2.5 Flash to extract receipt data directly to a structured JSON object.
    image_source can be a file path (str) or raw image bytes.
    """
    print(f"🔍 [Step 1] Sending image to Gemini 2.5 Flash...")
    
    # Load the image
    if isinstance(image_source, str):
        img = Image.open(image_source)
    else:
        import io
        img = Image.open(io.BytesIO(image_source))

    # Prompt for structured extraction
    prompt = "Analyze this receipt image and extract all purchased items along with the grand total."

    try:
        # Call model generate content using new SDK architecture
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img],
            config={
                'response_mime_type': 'application/json',
                'response_schema': ReceiptData,
            }
        )
        
        # New SDK parses response into .parsed attribute automatically if schema is provided
        if response.parsed:
            return response.parsed.model_dump()
        
        # Fallback in case parsing is skipped
        content = response.text.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        return json.loads(content)
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error during extraction: {error_msg}")
        if "503" in error_msg or "UNAVAILABLE" in error_msg or "high demand" in error_msg.lower():
            raise GeminiUnavailableError(
                "The extraction service is currently experiencing high demand. Please try again in a few moments."
            ) from e
        raise e

if __name__ == "__main__":
    # Test with a local file
    TEST_IMAGE = 'images/recibo_1.jpeg'
    
    if os.path.exists(TEST_IMAGE):
        result = extract_receipt_data(TEST_IMAGE)
        if result:
            print("\n✅ EXTRACTION SUCCESSFUL:\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ Failed to extract data.")
    else:
        print(f"❌ Test image {TEST_IMAGE} not found.")
