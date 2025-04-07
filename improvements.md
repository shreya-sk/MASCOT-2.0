# MASCOT-2.0: Comprehensive Documentation

## Getting Started

### Environment Setup
```bash
# Create and activate a virtual environment with Python 3.10
python3.10 -m venv senti
source senti/bin/activate

# Install required packages
pip install torch==2.0.1 transformers==4.36.0 wandb==0.15.12 spacy==3.7.2 scikit-learn==1.3.2 tqdm matplotlib seaborn

# Download spaCy model for syntax features
python -m spacy download en_core_web_sm
```

## Project Structure
Ensure your directory structure is as follows:
```
MASCOT-2.0/
├── Datasets/aste/
│   ├── laptop14/
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   └── test.txt
│   ├── rest14/
│   ├── rest15/
│   └── rest16/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── utils/
├── checkpoints/
├── train.py
├── test.py
└── predict.py
```

## Running the Training Pipeline

### Step 1: Set up your configuration
Review and adjust the configuration in `src/utils/config.py` based on your requirements:

```python
@dataclass
class StellaABSAConfig:
    # Model settings
    model_name: str = "stanford-crfm/Stella-400M-v5"   # or "stanford-crfm/Stella-400M-v5"
    hidden_size: int = 768  # Adjust based on available GPU memory
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 16  # Adjust based on GPU memory
    num_epochs: int = 10
    
    # Novel architecture components
    use_syntax: bool = True  # Enable syntax-guided attention
    use_aspect_first: bool = True  # Prioritize aspect over opinion
```

### Step 2: Basic Training
Run training on all datasets:
```bash
python train.py
```

Or train on a specific dataset:
```bash
python train.py --dataset rest15
```

The training script will automatically:
- Train the model using train splits
- Evaluate on dev splits periodically
- Save the best model based on validation F1 score
- Log metrics to Weights & Biases

### Step 3: Hyperparameter Tuning
```bash
# Run the sweep with our provided sweep configuration
python run_sweep.py
```

For hyperparameter tuning, the following parameters have the most impact:
- `learning_rate`: Try values between 1e-5 and 1e-4
- `batch_size`: 8, 16, or 32 (depending on GPU memory)
- `hidden_size`: 768 or 1024
- `dropout`: Try values between 0.1 and 0.3
- `num_layers`: 2, 3, or 4

### Step 4: Model Selection
After hyperparameter tuning, inspect the Weights & Biases dashboard to:
1. Identify the best performing hyperparameter configuration
2. Look for configurations that provide a good balance between performance and efficiency

Then train a final model with the best configuration:
```bash
python train.py --learning_rate 2e-5 --batch_size 16 --hidden_size 768 --dropout 0.2
```

## Validation and Testing

### Validation Process
- Validation occurs automatically during training
- The dev splits are used to:
  1. Calculate F1 scores for aspect detection, opinion detection, and sentiment classification
  2. Select the best performing model based on overall F1 score
  3. Guide early stopping and model selection

### Testing Process
Once your model is fully trained, evaluate it on the test sets:
```bash
python test.py --model checkpoints/experiment_name_rest15_best.pt
```

This will:
- Evaluate the model on all test sets (or a specific one if specified)
- Report detailed metrics including:
  - Precision, recall, and F1 scores for aspect and opinion extraction
  - Sentiment classification accuracy
  - Overall performance metrics

## Performance Enhancements

Within the current framework, consider these performance improvements:

### 1. Context-Aware Span Enhancement
Adjust the context window in the preprocessor to fine-tune span detection:
```python
# In src/data/preprocessor.py
preprocessor = StellaABSAPreprocessor(
    tokenizer=tokenizer,
    max_length=config.max_seq_length,
    use_syntax=config.use_syntax,
    context_window=3  # Increase from default 2
)
```

### 2. Attention Mechanism Improvements
Add an attention temperature parameter to control focus:
```python
# In src/models/cross_attention.py
attention_scores = attention_scores / self.temperature  # Add temperature parameter
```

### 3. Loss Weighting for Imbalanced Classes
Adjust loss weights based on class distributions:
```python
# In config.py
aspect_loss_weight: float = 1.2  # Increase if aspect detection is challenging
opinion_loss_weight: float = 1.0
sentiment_loss_weight: float = 0.8
```

### 4. Mixed Precision Training
Ensure mixed precision is enabled for faster training:
```python
# In config.py
use_fp16: bool = True
```

### 5. Learning Rate Scheduling
Experiment with different learning rate schedulers:
```python
# In train.py, replace linear schedule with cosine
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

## Novelty Enhancements

To increase the research contribution without major code refactoring:

### 1. Generative ABSA Integration

Integrate generative capabilities by adding a decoder that converts extracted triplets into natural language:

```python
# Add to src/models/absa.py
class GenerativeOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4
        )
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, hidden_states, aspect_logits, opinion_logits, sentiment_logits):
        # Generate text summaries of the ABSA results
        decoder_output = self.decoder(
            hidden_states, 
            self._create_memory_from_triplets(aspect_logits, opinion_logits, sentiment_logits)
        )
        return self.linear(decoder_output)
```

### 2. Prompt Engineering for Structured Output

Restructure the input for generative outputs:

```python
def generate_structured_output(text, triplets):
    """Transform extracted triplets into a structured natural language format"""
    template = "In the text '{text}', I found that {aspect} is {sentiment} because of {opinion}."
    
    outputs = []
    for triplet in triplets:
        aspect = triplet['aspect']
        opinion = triplet['opinion']
        sentiment = "positive" if triplet['sentiment'] == "POS" else \
                    "negative" if triplet['sentiment'] == "NEG" else "neutral"
        
        outputs.append(template.format(
            text=text,
            aspect=aspect,
            sentiment=sentiment,
            opinion=opinion
        ))
    
    return outputs
```

### 3. Contrastive Learning Integration

Implement contrastive learning to improve representations:

```python
# Add to src/training/losses.py
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, aspect_features, opinion_features, sentiment_labels):
        # Create positive and negative pairs based on sentiment
        # This enhances the relationship modeling between aspects and opinions
        normalized_aspects = F.normalize(aspect_features, dim=-1)
        normalized_opinions = F.normalize(opinion_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(normalized_aspects, normalized_opinions.transpose(0, 1))
        similarity = similarity / self.temperature
        
        # Create target matrix (1 for same sentiment, 0 otherwise)
        targets = (sentiment_labels.unsqueeze(0) == sentiment_labels.unsqueeze(1)).float()
        
        # Compute contrastive loss
        loss = F.binary_cross_entropy_with_logits(similarity, targets)
        return loss
```

### 4. Key Insight Extraction

Add a module for extracting key insights from the identified triplets:

```python
def extract_key_insights(triplets, min_confidence=0.8):
    """Extract key insights from ABSA triplets"""
    # Group by sentiment
    sentiments = {"POS": [], "NEG": [], "NEU": []}
    
    for triplet in triplets:
        if triplet['confidence'] >= min_confidence:
            sentiments[triplet['sentiment']].append(triplet)
    
    insights = []
    
    # Extract positive insights
    if sentiments["POS"]:
        top_positive = max(sentiments["POS"], key=lambda x: x['confidence'])
        insights.append(f"The most positive aspect is '{top_positive['aspect']}' due to '{top_positive['opinion']}'.")
    
    # Extract negative insights
    if sentiments["NEG"]:
        top_negative = max(sentiments["NEG"], key=lambda x: x['confidence'])
        insights.append(f"The main concern is '{top_negative['aspect']}' because it is '{top_negative['opinion']}'.")
    
    # Extract overall sentiment
    sentiment_counts = {k: len(v) for k, v in sentiments.items()}
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    
    sentiment_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
    insights.append(f"The overall sentiment is {sentiment_map[dominant_sentiment]}.")
    
    return insights
```

## Publication Strategy

To maximize the chances of publication:

### Key Contributions to Highlight

1. **Syntax-Guided Attention**: Emphasize how incorporating syntactic structure improves boundary detection for aspect and opinion terms.

2. **Context-Aware Span Detection**: Highlight the novel bidirectional influence between aspects and opinions.

3. **Aspect-Opinion Joint Learning**: Stress how the model learns the interdependence between aspects and opinions.

4. **Generative ABSA**: Showcase the model's ability to generate structured natural language summaries from extracted triplets.

5. **Multi-Domain Knowledge Transfer**: Explain how the model enables knowledge transfer across domains (restaurant, laptop).

### Ablation Studies

Include comprehensive ablation studies:

1. **Component Impact**: Remove each novel component one by one to measure its contribution.
2. **Syntax Importance**: Compare performance with and without syntax-guided attention.
3. **Context Window Size**: Vary the context window size to show its impact on performance.
4. **Embedding Comparisons**: Compare Stella vs. LLAMA embeddings.

### Comparison to SOTA

Ensure your benchmark includes:
- Traditional ABSA methods (LSTM, BERT-based)
- Recent LLM-based methods
- Zero-shot/few-shot prompting approaches

## Conclusion

By following this comprehensive guide, you should be able to effectively train, validate, and test the MASCOT-2.0 model, as well as introduce novel enhancements that will strengthen your research contribution. The suggested improvements maintain the current architecture while adding significant novelty through generative capabilities and structured insight extraction.

For publication success, focus on thorough experimentation, clear ablation studies, and comparison to the state-of-the-art, while emphasizing the unique aspects of your approach in combining structured and generative ABSA.