# test_instruct.py
import torch
from transformers import AutoTokenizer
from src.utils.config import LLMABSAConfig
from src.models.absa import LLMABSA
from src.models.instruct_absa_minimal import MinimalInstructABSA

def test_instruct_absa():
    print("🧪 Testing InstructABSA integration...")
    
    # Load config and models
    config = LLMABSAConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Create models
    existing_model = LLMABSA(config)
    instruct_model = MinimalInstructABSA(config, existing_model)
    
    # Test input
    test_text = "The food was delicious but the service was terrible"
    inputs = tokenizer(test_text, return_tensors='pt', max_length=128, 
                      padding='max_length', truncation=True)
    
    print(f"Input: {test_text}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = instruct_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            task_type='triplet_extraction'
        )
    
    print("✅ Forward pass successful!")
    print(f"Generated text: {outputs['generated_text']}")
    print(f"Extracted triplets: {outputs['extracted_triplets']}")
    
    # Test with target (training mode)
    target_text = "<triplet><aspect>food</aspect><opinion>delicious</opinion><sentiment>POS</sentiment></triplet> <triplet><aspect>service</aspect><opinion>terrible</opinion><sentiment>NEG</sentiment></triplet>"
    
    train_outputs = instruct_model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        task_type='triplet_extraction',
        target_text=target_text
    )
    
    print("✅ Training mode successful!")
    print(f"Loss: {train_outputs['loss'].item():.4f}")

if __name__ == '__main__':
    test_instruct_absa()