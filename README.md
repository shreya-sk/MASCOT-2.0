
# MASCOT-2.0 with Stella v5

An advanced Aspect-Based Sentiment Analysis (ABSA) model using Stella v5 (400M), a light-weight, high-performance language model that delivers superior performance for ABSA tasks.

## Overview

This project implements a novel ABSA approach using the open-source Stella v5 (400M) model with several innovative components:

1. **Hierarchical Focal Embedding**: A dual-projection architecture that handles aspect and opinion terms separately.
2. **Context-Aware Span Detection**: Novel bidirectional modeling between aspects and opinions.
3. **Syntax-Guided Attention**: Incorporates syntactic structure for improved boundary detection.
4. **Aspect-Opinion Joint Classification**: Simultaneous consideration of aspects and opinions for sentiment analysis.
5. **Multi-Domain Knowledge Transfer**: Cross-domain adaptation for improved performance across domains.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mascot-2.0.git
cd mascot-2.0

# Create and activate virtual environment
python -m venv senti
source senti/bin/activate  # On Windows: senti\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for syntax features
python -m spacy download en_core_web_sm
```

## Directory Structure

```
MASCOT-2.0/
├── checkpoints/                  # Model checkpoints
├── Dataset/
│   └── aste/
│       ├── laptop14/
│       ├── rest14/
│       ├── rest15/
│       └── rest16/               # ABSA datasets
├── src/
│   ├── data/                     # Data processing
│   │   ├── dataset.py
│   │   ├── stella_preprocessor.py
│   │   └── utils.py
│   ├── inference/                # Inference pipeline
│   │   └── stella_predictor.py
│   ├── models/                   # Model architecture
│   │   ├── aspect_opinion_joint_classifier.py
│   │   ├── context_span_detector.py
│   │   ├── stella_absa.py
│   │   └── stella_embedding.py
│   ├── training/                 # Training components
│   │   ├── losses.py
│   │   └── metrics.py
│   └── utils/                    # Utilities
│       ├── logger.py
│       ├── stella_config.py
│       └── visualization.py
├── train_stella.py               # Training script
├── predict_stella.py             # Prediction script
└── requirements.txt
```

## Key Innovations

### 1. Hierarchical Focal Embedding

Our model uses a hierarchical focal embedding approach with Stella v5 (400M) that creates separate specialized projections for aspect terms and opinion terms. This allows the model to better distinguish between these different semantic roles.

```python
aspect_embeddings = self.aspect_projection(hidden_states)
opinion_embeddings = self.opinion_projection(hidden_states)
```

### 2. Syntax-Guided Attention

We incorporate syntactic information to improve span boundary detection using a novel syntax-guided attention mechanism:

```python
# Dynamic span-aware attention weights with syntax
attention_scores = attention_scores + syntax_attention
```

### 3. Bidirectional Aspect-Opinion Influence

Our model implements a novel bidirectional influence mechanism where aspect representations affect opinion detection and vice versa:

```python
# Bidirectional influence
aspect_to_opinion_influence = self.aspect_to_opinion(aspect_attn)
opinion_to_aspect_influence = self.opinion_to_aspect(opinion_attn)

# Enhanced representations
enhanced_aspect = aspect_attn + opinion_to_aspect_influence
enhanced_opinion = opinion_attn + aspect_to_opinion_influence
```

### 4. Aspect-Opinion Joint Classification

We use a novel joint classifier that simultaneously considers both aspect and opinion representations, modeling their interdependencies:

```python
# Novel triple attention for complex interactions
aspect_attn, opinion_attn, context_attn = self.triple_attention(...)

# Adaptive fusion based on aspect-opinion interaction
fusion_weights = self.fusion_gate(fusion_input)
```

### 5. Cross-Domain Knowledge Transfer

Our model implements domain adaptation that allows knowledge transfer across different domains (e.g., restaurant reviews to laptop reviews):

```python
# Apply domain adaptation if domain is specified
if domain_id is not None:
    domain_embeddings = self.domain_adapter(hidden_states)
    hidden_states = hidden_states + domain_embeddings
```

## Model Training

To train the model:

```bash
python train_stella.py --dataset rest15
```

Options:
- `--dataset`: Specific dataset to train on
- `--device`: Device to use (cuda or cpu)
- `--seed`: Random seed for reproducibility

## Making Predictions

To run predictions with a trained model:

```bash
python predict_stella.py --model checkpoints/stella-absa-v5_rest15_best.pt --text "The food was delicious but the service was terrible." --visualize
```

Options:
- `--model`: Path to model checkpoint
- `--text`: Text to analyze
- `--file`: File with texts to analyze (one per line)
- `--output`: Output file for results
- `--visualize`: Create HTML visualization

## Example Predictions

Input: "The food was delicious but the service was terrible."

Output:
```
Predictions:
  Aspect: food, Opinion: delicious, Sentiment: POS (Confidence: 0.92)
  Aspect: service, Opinion: terrible, Sentiment: NEG (Confidence: 0.94)
```

## Acknowledgments

- Stanford CRFM for the Stella v5 model
- Authors of the original ABSA datasets
- The open-source NLP community

## License

MIT License
