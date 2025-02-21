# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List

class AttentionVisualizer:
    """Visualize attention patterns and model predictions"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def plot_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        save_path: str = None
    ):
        """Plot attention heatmap"""
        # Convert attention weights to numpy
        attn = attention_weights.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd'
        )
        
        plt.title("Attention Weights")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def visualize_spans(
        self,
        text: str,
        aspect_spans: List[tuple],
        opinion_spans: List[tuple],
        save_path: str = None
    ):
        """Visualize detected spans in text"""
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        # Create visualization
        colors = {"aspect": "#fbd971", "opinion": "#b8e994"}
        options = {"ents": ["aspect", "opinion"], "colors": colors}
        
        spacy.displacy.render(doc, style="ent", options=options)
        
        if save_path:
            svg = spacy.displacy.render(doc, style="ent", options=options)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(svg)