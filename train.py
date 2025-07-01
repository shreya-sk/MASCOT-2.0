import torch
from transformers import AutoTokenizer
from src.utils.config import LLMABSAConfig
from src.models.absa import LLMABSA
from src.training.losses import ABSALoss

def create_training_examples():
    """Create some training examples for instruction-following"""
    examples = [
        {
            'text': "The food was delicious but the service was terrible",
            'target': "<triplet><aspect>food</aspect><opinion>delicious</opinion><sentiment>POS</sentiment></triplet> <triplet><aspect>service</aspect><opinion>terrible</opinion><sentiment>NEG</sentiment></triplet>"
        },
        {
            'text': "Great pizza and fast delivery",
            'target': "<triplet><aspect>pizza</aspect><opinion>great</opinion><sentiment>POS</sentiment></triplet> <triplet><aspect>delivery</aspect><opinion>fast</opinion><sentiment>POS</sentiment></triplet>"
        },
        {
            'text': "The pasta was okay but overpriced",
            'target': "<triplet><aspect>pasta</aspect><opinion>okay</opinion><sentiment>NEU</sentiment></triplet> <triplet><aspect>pasta</aspect><opinion>overpriced</opinion><sentiment>NEG</sentiment></triplet>"
        },
        {
            'text': "Excellent ambiance and friendly staff",
            'target': "<triplet><aspect>ambiance</aspect><opinion>excellent</opinion><sentiment>POS</sentiment></triplet> <triplet><aspect>staff</aspect><opinion>friendly</opinion><sentiment>POS</sentiment></triplet>"
        },
        {
            'text': "The burger was cold and the fries were soggy",
            'target': "<triplet><aspect>burger</aspect><opinion>cold</opinion><sentiment>NEG</sentiment></triplet> <triplet><aspect>fries</aspect><opinion>soggy</opinion><sentiment>NEG</sentiment></triplet>"
        }
    ]
    return examples

def train_instruction_component():
    print("🚀 Training instruction-following component...")
    
    # Load config and model
    config = LLMABSAConfig()
    config.learning_rate = 1e-4  # Higher learning rate for T5
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = LLMABSA(config)
    
    # Create optimizer for T5 components only
    t5_params = list(model.t5_model.parameters()) + list(model.feature_bridge.parameters())
    optimizer = torch.optim.AdamW(t5_params, lr=config.learning_rate)
    
    # Create loss function
    loss_fn = ABSALoss(config)
    
    # Get training examples
    examples = create_training_examples()
    
    print(f"Training on {len(examples)} examples for 20 steps...")
    
    model.train()
    for epoch in range(4):  # 4 epochs
        total_loss = 0
        for i, example in enumerate(examples):
            # Tokenize input
            inputs = tokenizer(
                example['text'], 
                return_tensors='pt', 
                max_length=128, 
                padding='max_length', 
                truncation=True
            )
            
            # Forward pass with target text
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                texts=[example['text']],
                task_type='triplet_extraction',
                target_text=example['target']
            )
            
            # Get generation loss
            if 'generation_loss' in outputs:
                loss = outputs['generation_loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                print(f"Epoch {epoch+1}, Example {i+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(examples)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    print("✅ Training completed!")
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    model.eval()
    
    test_text = "The sushi was fresh but the restaurant was noisy"
    inputs = tokenizer(test_text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            texts=[test_text],
            task_type='triplet_extraction'
        )
    
    if 'generated_text' in outputs:
        print(f"Input: {test_text}")
        print(f"Generated: {outputs['generated_text']}")
        
        # Parse the generated triplets
        if hasattr(model, '_parse_generated_triplets'):
            triplets = model._parse_generated_triplets(outputs['generated_text'])
            print(f"Parsed triplets: {triplets}")
    
    return model

if __name__ == '__main__':
    trained_model = train_instruction_component()