# tests/test_utils.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import calculate_confidence, format_response, validate_file

print("Testing src/utils.py...")

# Test 1: Confidence calculation
scores = [0.85, 0.82, 0.78]
confidence = calculate_confidence(scores)
print(f"✓ Confidence (scores {scores}): {confidence}")
assert confidence == "High", "Should be High confidence"

scores = [0.65, 0.62]
confidence = calculate_confidence(scores)
print(f"✓ Confidence (scores {scores}): {confidence}")
assert confidence == "Medium", "Should be Medium confidence"

# Test 2: Response formatting
answer = "This is a test answer"
sources = "doc1.pdf (page 1)\ndoc2.pdf (page 3)"  # Fixed: sources should be string
confidence = "High"
formatted = format_response(answer, sources, confidence)
print(f"✓ Formatted response length: {len(formatted)} chars")
assert "test answer" in formatted.lower(), "Answer should be in formatted response"

# Test 3: File validation
# Create a dummy file
test_file = "test_dummy.txt"
with open(test_file, "w") as f:
    f.write("Test content")

is_valid, error = validate_file(test_file)
print(f"✓ File validation: {is_valid}, Error: {error}")
assert is_valid == True, "Valid file should pass"

# Clean up
os.remove(test_file)

print("✅ All utils tests passed!")