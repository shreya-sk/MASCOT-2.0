#!/usr/bin/env python
# diagnosis.py - Script to diagnose model issues
import argparse
import torch
from transformers import AutoTokenizer

from src.utils.config import LLMABSAConfig
from src.models.absa import LLMABSA

def diagnose_model(model_path, text="The food was delicious but the service was slow.", device=None):
    """Run a diagnosis on the ABSA model to identify issues"""
    print(f"=== ABSA Model Diagnosis for {model_path} ===\n")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load configuration
    config = LLMABSAConfig()
    print(f"Config loaded: hidden_size={config.hidden_size}, model_name={config.model_name}")
    
    # Load model with detailed error tracking
    try:
        print("\nAttempting to load model...")
        model = LLMABSA.load(model_path, config=config, device=device)
        print("Model loaded successfully")
        
        # Print model architecture
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,} ({num_trainable:,} trainable)")
        
        # Verify components
        print("\nChecking model components:")
        if hasattr(model, 'embeddings'):
            print("✓ Embeddings module found")
        else:
            print("✗ Embeddings module missing")
            
        if hasattr(model, 'span_detector'):
            print("✓ Span detector module found")
        else:
            print("✗ Span detector module missing")
            
        if hasattr(model, 'sentiment_classifier'):
            print("✓ Sentiment classifier module found")
        else:
            print("✗ Sentiment classifier module missing")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load tokenizer
    try:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test text encoding
    try:
        print("\nTesting text encoding...")
        inputs = tokenizer(text, return_tensors="pt").to(device)
        print(f"Text encoded: {inputs.keys()}")
        print(f"input_ids shape: {inputs['input_ids'].shape}")
    except Exception as e:
        print(f"Error encoding text: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model forward pass
    try:
        print("\nRunning model forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("Forward pass completed")
        print(f"Output keys: {outputs.keys()}")
        
        if 'aspect_logits' in outputs:
            print(f"✓ aspect_logits found: {outputs['aspect_logits'].shape}")
        else:
            print("✗ aspect_logits missing")
            
        if 'opinion_logits' in outputs:
            print(f"✓ opinion_logits found: {outputs['opinion_logits'].shape}")
        else:
            print("✗ opinion_logits missing")
            
        if 'sentiment_logits' in outputs:
            print(f"✓ sentiment_logits found: {outputs['sentiment_logits'].shape}")
            # Show sentiment distribution
            probs = torch.softmax(outputs['sentiment_logits'], dim=-1)
            print(f"  Sentiment distribution: POS={probs[0,0].item():.2f}, NEU={probs[0,1].item():.2f}, NEG={probs[0,2].item():.2f}")
        else:
            print("✗ sentiment_logits missing")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model triplet extraction
    try:
        print("\nTesting triplet extraction...")
        with torch.no_grad():
            triplets = model.extract_triplets(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                tokenizer=tokenizer,
                texts=[text]
            )
        
        print("Triplet extraction completed")
        if triplets and triplets[0]:
            print(f"Found {len(triplets[0])} triplets")
            for i, t in enumerate(triplets[0]):
                print(f"  Triplet {i+1}: aspect='{t.get('aspect', '')}', opinion='{t.get('opinion', '')}', sentiment={t.get('sentiment', '')}")
        else:
            print("No triplets found")
    except Exception as e:
        print(f"Error in triplet extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Diagnosis Complete ===")

def main():
    parser = argparse.ArgumentParser(description='Diagnose ABSA model issues')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default="The food was delicious but the service was slow.", help='Text to analyze')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    diagnose_model(args.model, args.text, args.device)

if __name__ == '__main__':
    main()