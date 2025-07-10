# test_instruction_improved.py
import torch
from transformers import AutoTokenizer
from src.utils.config import LLMGRADIENTConfig
from src.models.model import GRADIENTModel

def test_improved_instruction():
    print("🧪 Testing improved InstructABSA...")
    
    config = LLMGRADIENTConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = GRADIENTModel(config)
    
    # Test cases
    test_cases = [
        "The food was delicious but the service was terrible",
        "Great pizza and excellent service",
        "The pasta was overpriced but tasty",
        "Terrible food and rude staff"
    ]
    
    for test_text in test_cases:
        print(f"\n📝 Input: {test_text}")
        
        inputs = tokenizer(test_text, return_tensors='pt', max_length=128, 
                          padding='max_length', truncation=True)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                texts=[test_text],
                task_type='triplet_extraction'
            )
        
        if 'generated_text' in outputs:
            print(f"🤖 Generated: {outputs['generated_text']}")
            
            # Try to parse triplets
            if hasattr(model, '_parse_generated_triplets'):
                triplets = model._parse_generated_triplets(outputs['generated_text'])
                if triplets:
                    print(f"✅ Parsed triplets: {triplets}")
                else:
                    print("⚠ No triplets parsed from generated text")

if __name__ == '__main__':
    test_improved_instruction()