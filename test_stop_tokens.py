#!/usr/bin/env python3
"""
Test script to validate stop token fix for VLM calls.
This compares different stop token configurations to identify the root cause.
"""
import base64
import sys
from pathlib import Path
from openai import OpenAI

# Configuration
VLM_SERVER = "http://192.168.68.186:8080/v1"
MODEL_NAME = "Qwen2.5VL-3B"
TEST_IMAGE = "test_page.jpg"  # User should provide a test image

def load_test_image(image_path):
    """Load and encode test image as base64."""
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
            return base64.b64encode(img_bytes).decode("utf-8")
    except FileNotFoundError:
        print(f"❌ Error: Test image not found at {image_path}")
        print("Please provide a test PDF page image as test_page.jpg")
        sys.exit(1)

def create_messages(img_b64, prompt="Transcribe this PDF page to clean Markdown."):
    """Create the messages payload for the VLM API."""
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }]

def test_with_stop_config(client, messages, stop_config, test_name):
    """Run a single test with specific stop token configuration."""
    print("=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)
    
    try:
        params = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        
        # Add stop parameter based on config
        if stop_config is not None:
            params["stop"] = stop_config
            print(f"Stop tokens: {stop_config}")
        else:
            print("Stop tokens: (not specified, using client defaults)")
        
        # Make API call
        res = client.chat.completions.create(**params)
        
        # Extract results
        content = res.choices[0].message.content
        tokens = res.usage.total_tokens
        finish_reason = res.choices[0].finish_reason
        
        # Print results
        print(f"\n✅ SUCCESS")
        print(f"Tokens used: {tokens}")
        print(f"Finish reason: {finish_reason}")
        print(f"Content length: {len(content)} characters")
        print(f"\nFirst 200 characters:")
        print("-" * 80)
        print(content[:200])
        print("-" * 80)
        
        # Verdict
        if tokens < 50:
            print(f"\n⚠️  WARNING: Very low token count ({tokens}). Likely premature termination!")
        elif len(content) < 100:
            print(f"\n⚠️  WARNING: Very short response ({len(content)} chars). Possible issue!")
        else:
            print(f"\n✅ GOOD: Response looks reasonable ({tokens} tokens, {len(content)} chars)")
        
        return {
            "success": True,
            "tokens": tokens,
            "finish_reason": finish_reason,
            "length": len(content),
            "content": content
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Run all stop token tests."""
    print("\n" + "=" * 80)
    print("VLM STOP TOKEN TEST SUITE")
    print("=" * 80)
    print(f"Server: {VLM_SERVER}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 80)
    
    # Load test image
    img_b64 = load_test_image(TEST_IMAGE)
    messages = create_messages(img_b64)
    
    # Create OpenAI client
    client = OpenAI(
        api_key="not-needed",
        base_url=VLM_SERVER
    )
    
    # Run tests
    results = {}
    
    # Test 1: With stop=[] (EXPECTED TO WORK)
    results["empty_stop"] = test_with_stop_config(
        client, messages, [], 
        "With stop=[] (Explicitly disable stop tokens)"
    )
    print("\n")
    
    # Test 2: Without stop parameter (EXPECTED TO FAIL)
    results["no_stop"] = test_with_stop_config(
        client, messages, None,
        "Without stop parameter (Client defaults)"
    )
    print("\n")
    
    # Test 3: With OpenAI-style stop tokens (EXPECTED TO FAIL)
    results["openai_stop"] = test_with_stop_config(
        client, messages, ["<|im_end|>", "\n\n"],
        "With OpenAI-style stop tokens"
    )
    print("\n")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result["success"]:
            status = "✅ PASS" if result["tokens"] > 50 else "⚠️  SUSPICIOUS"
            print(f"{status} | {test_name:20s} | {result['tokens']:4d} tokens | {result['length']:5d} chars")
        else:
            print(f"❌ FAIL | {test_name:20s} | Error: {result.get('error', 'Unknown')}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    empty_stop = results.get("empty_stop", {})
    no_stop = results.get("no_stop", {})
    
    if empty_stop.get("success") and empty_stop.get("tokens", 0) > 100:
        print("✅ GOOD: stop=[] produces long responses")
        print("   → This confirms the fix works!")
        
        if no_stop.get("success") and no_stop.get("tokens", 0) < 50:
            print("\n⚠️  CONFIRMED: Without stop=[], OpenAI client adds default stop tokens")
            print("   → RAGFlow MUST use stop=[] to prevent premature termination")
        else:
            print("\n✅ Interestingly, no stop parameter also works")
            print("   → May depend on OpenAI client version or server config")
    else:
        print("❌ PROBLEM: Even with stop=[], responses are short")
        print("   → Issue may not be stop tokens. Check:")
        print("     1. Prompt wording (does it suggest stopping early?)")
        print("     2. Image format (is it valid?)")
        print("     3. Server configuration")
        print("     4. Model compatibility")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()