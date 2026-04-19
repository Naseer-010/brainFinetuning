"""Quick test of Ollama vision models on the PDFs."""
import fitz, base64, requests, json, time

def test_vision(pdf_path, page_num, model="qwen2.5vl:7b"):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(1.5, 1.5)
    pix = page.get_pixmap(matrix=mat)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    print(f"Image b64 length: {len(img_b64)} chars")

    url = "http://localhost:11434/api/chat"
    
    prompt = (
        "Extract ALL questions from this JEE exam page as a JSON array. "
        "For each question output: "
        '{"question_number": <int>, "text": "<full question with all options A/B/C/D or 1/2/3/4>", '
        '"subject": "<physics|chemistry|math>", "has_diagram": <true/false>}. '
        "Output ONLY a valid JSON array, nothing else."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You extract structured data from exam papers. Output only valid JSON."},
            {"role": "user", "content": prompt, "images": [img_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 4096},
        "format": "json",
    }
    
    print(f"Calling {model}...")
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=300)
        elapsed = time.time() - start
        print(f"Status: {resp.status_code}, Time: {elapsed:.1f}s")
        result = resp.json()
        content = result.get("message", {}).get("content", "")
        print(f"Response length: {len(content)}")
        print(f"Response:\n{content[:3000]}")
        
        # Try parsing
        try:
            parsed = json.loads(content)
            print(f"\nJSON parsed OK! Type: {type(parsed).__name__}")
            if isinstance(parsed, list):
                print(f"Items: {len(parsed)}")
            elif isinstance(parsed, dict):
                print(f"Keys: {list(parsed.keys())}")
                if "questions" in parsed:
                    print(f"Questions: {len(parsed['questions'])}")
        except json.JSONDecodeError as e:
            print(f"\nJSON parse error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    doc.close()

# Test 1: Chaitanya question page (math, page 3)
print("=" * 60)
print("TEST 1: Chaitanya Questions - Page 3 (Math)")
print("=" * 60)
test_vision("data/questions/chaitanya_1.pdf", 2)

# Test 2: Narayana question page (physics, page 1)
print("\n" + "=" * 60)
print("TEST 2: Narayana Questions - Page 1 (Physics)")
print("=" * 60)
test_vision("data/questions/narayana_1.pdf", 0)

# Test 3: Chaitanya solution key page
print("\n" + "=" * 60)
print("TEST 3: Chaitanya Solution Key - Page 1")
print("=" * 60)

doc = fitz.open("data/solutions/chaitanya_key_1.pdf")
page = doc[0]
mat = fitz.Matrix(1.5, 1.5)
pix = page.get_pixmap(matrix=mat)
img_b64 = base64.b64encode(pix.tobytes("png")).decode()

url = "http://localhost:11434/api/chat"
key_prompt = (
    "This is an answer key sheet. Extract ALL question number and answer mappings as a JSON array. "
    'Each item: {"question_number": <int>, "answer": "<answer value>", "subject": "<math|physics|chemistry>"}. '
    "Output ONLY a valid JSON array."
)

payload = {
    "model": "qwen2.5vl:7b",
    "messages": [
        {"role": "system", "content": "You extract answer keys from exam papers. Output only valid JSON."},
        {"role": "user", "content": key_prompt, "images": [img_b64]},
    ],
    "stream": False,
    "options": {"temperature": 0.1, "num_predict": 4096},
    "format": "json",
}

start = time.time()
try:
    resp = requests.post(url, json=payload, timeout=300)
    elapsed = time.time() - start
    print(f"Status: {resp.status_code}, Time: {elapsed:.1f}s")
    result = resp.json()
    content = result.get("message", {}).get("content", "")
    print(f"Response length: {len(content)}")
    print(f"Response:\n{content[:3000]}")
    parsed = json.loads(content)
    print(f"\nJSON parsed OK! Type: {type(parsed).__name__}")
except Exception as e:
    print(f"Error: {e}")
doc.close()
