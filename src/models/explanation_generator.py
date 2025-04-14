# src/models/explanation_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ExplanationGenerator:
    """
    Simple template-based explanation generator for ABSA triplets
    
    This class creates natural language explanations from extracted triplets
    without requiring a separate neural model, making it very efficient.
    """
    def __init__(self):
        # Templates for different sentiment types
        self.positive_templates = [
            "The {aspect} is positive because of its {opinion}.",
            "I liked the {aspect} due to its {opinion}.",
            "The {aspect} was great, especially the {opinion}.",
            "The {opinion} of the {aspect} was really good.",
            "Thanks to its {opinion}, the {aspect} was quite enjoyable."
        ]
        
        self.negative_templates = [
            "The {aspect} is negative because of its {opinion}.",
            "I didn't like the {aspect} due to its {opinion}.",
            "The {aspect} was disappointing, especially the {opinion}.",
            "The {opinion} of the {aspect} was really poor.",
            "Because of its {opinion}, the {aspect} was not enjoyable."
        ]
        
        self.neutral_templates = [
            "The {aspect} is neutral with {opinion}.",
            "The {aspect} was neither good nor bad with its {opinion}.",
            "The {opinion} of the {aspect} was just okay.",
            "The {aspect} had average {opinion}.",
            "The {aspect} with its {opinion} was standard, nothing special."
        ]
        
        # Templates for empty opinions
        self.aspect_only_templates = {
            'POS': [
                "The {aspect} was excellent.",
                "I really enjoyed the {aspect}.",
                "The {aspect} was one of the highlights."
            ],
            'NEG': [
                "The {aspect} was disappointing.",
                "I didn't like the {aspect}.",
                "The {aspect} was a letdown."
            ],
            'NEU': [
                "The {aspect} was average.",
                "The {aspect} was neither good nor bad.",
                "The {aspect} was just okay."
            ]
        }
        
        # Templates for combining multiple triplets
        self.combo_templates = [
            "While the {pos_aspect} was excellent with its {pos_opinion}, the {neg_aspect} was disappointing due to {neg_opinion}.",
            "I enjoyed the {pos_aspect} because of its {pos_opinion}, but the {neg_aspect} could be improved as it was {neg_opinion}.",
            "The {pos_aspect} had great {pos_opinion}, though the {neg_aspect} was lacking with its {neg_opinion}.",
            "Despite the excellent {pos_opinion} of the {pos_aspect}, the {neg_aspect} was let down by its {neg_opinion}."
        ]
    
    def generate_explanation(self, triplets):
        """
        Generate a natural language explanation from triplets
        
        Args:
            triplets: List of triplet dictionaries with aspect, opinion, and sentiment
            
        Returns:
            String containing the natural language explanation
        """
        if not triplets:
            return "No aspects were detected in this review."
        
        # If only one triplet, use simple template
        if len(triplets) == 1:
            return self._generate_single_explanation(triplets[0])
        
        # Group triplets by sentiment
        positive_triplets = [t for t in triplets if t['sentiment'] == 'POS']
        negative_triplets = [t for t in triplets if t['sentiment'] == 'NEG']
        neutral_triplets = [t for t in triplets if t['sentiment'] == 'NEU']
        
        # Generate combined explanation based on sentiment distribution
        if positive_triplets and negative_triplets:
            # Mixed sentiment - generate contrast explanation
            return self._generate_contrast_explanation(positive_triplets, negative_triplets)
        elif len(positive_triplets) > 1:
            # Multiple positive aspects
            return self._generate_multi_sentiment_explanation(positive_triplets, "positive")
        elif len(negative_triplets) > 1:
            # Multiple negative aspects
            return self._generate_multi_sentiment_explanation(negative_triplets, "negative")
        elif len(neutral_triplets) > 1:
            # Multiple neutral aspects
            return self._generate_multi_sentiment_explanation(neutral_triplets, "neutral")
        else:
            # Fallback to individual explanations
            explanations = []
            for triplet in triplets:
                explanations.append(self._generate_single_explanation(triplet))
            return " ".join(explanations)
    
    def _generate_single_explanation(self, triplet):
        """Generate explanation for a single triplet"""
        aspect = triplet.get('aspect', '').strip()
        opinion = triplet.get('opinion', '').strip()
        sentiment = triplet.get('sentiment', 'NEU')
        
        # Handle empty fields
        if not aspect:
            aspect = "this"
        
        # If opinion is empty, use aspect-only template
        if not opinion:
            templates = self.aspect_only_templates.get(sentiment, self.aspect_only_templates['NEU'])
            template_idx = hash(aspect) % len(templates)  # Use hash for consistent but varied selection
            template = templates[template_idx]
            return template.format(aspect=aspect)
        
        # Select template based on sentiment
        if sentiment == 'POS':
            templates = self.positive_templates
        elif sentiment == 'NEG':
            templates = self.negative_templates
        else:
            templates = self.neutral_templates
        
        # Select template based on hash of aspect for variety but consistency
        template_idx = hash(aspect + opinion) % len(templates)
        template = templates[template_idx]
        
        # Format template with aspect and opinion
        return template.format(aspect=aspect, opinion=opinion)
    
    def _generate_contrast_explanation(self, positive_triplets, negative_triplets):
        """Generate explanation contrasting positive and negative aspects"""
        # Sort by confidence to get the best triplets
        pos_triplet = sorted(positive_triplets, key=lambda x: x.get('confidence', 0.0), reverse=True)[0]
        neg_triplet = sorted(negative_triplets, key=lambda x: x.get('confidence', 0.0), reverse=True)[0]
        
        pos_aspect = pos_triplet.get('aspect', 'this').strip() or "this"
        pos_opinion = pos_triplet.get('opinion', '').strip() or "good quality"
        
        neg_aspect = neg_triplet.get('aspect', 'this').strip() or "this"
        neg_opinion = neg_triplet.get('opinion', '').strip() or "poor quality"
        
        # Avoid repetition if aspects are the same
        if pos_aspect.lower() == neg_aspect.lower():
            return f"The {pos_aspect} had both good and bad qualities - the {pos_opinion} was excellent, but the {neg_opinion} was disappointing."
        
        # Select contrast template
        template_idx = hash(pos_aspect + neg_aspect) % len(self.combo_templates)
        template = self.combo_templates[template_idx]
        
        # Format template
        return template.format(
            pos_aspect=pos_aspect,
            pos_opinion=pos_opinion,
            neg_aspect=neg_aspect,
            neg_opinion=neg_opinion
        )
    
    def _generate_multi_sentiment_explanation(self, triplets, sentiment_type):
        """Generate explanation for multiple aspects with the same sentiment"""
        # Sort by confidence
        sorted_triplets = sorted(triplets, key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Take top 2-3 triplets
        top_triplets = sorted_triplets[:min(3, len(sorted_triplets))]
        
        aspects = []
        opinions = []
        
        for triplet in top_triplets:
            aspect = triplet.get('aspect', '').strip()
            opinion = triplet.get('opinion', '').strip()
            
            if aspect:
                aspects.append(aspect)
            
            if opinion:
                opinions.append(opinion)
        
        # Filter duplicates
        aspects = list(dict.fromkeys(aspects))
        opinions = list(dict.fromkeys(opinions))
        
        # Generate explanation based on available info
        if not aspects:
            aspects = ["this"]
            
        if sentiment_type == "positive":
            if opinions:
                return f"The {', '.join(aspects[:-1]) + ' and ' + aspects[-1] if len(aspects) > 1 else aspects[0]} {'were' if len(aspects) > 1 else 'was'} excellent, particularly because of the {', '.join(opinions[:-1]) + ' and ' + opinions[-1] if len(opinions) > 1 else opinions[0]}."
            else:
                return f"The {', '.join(aspects[:-1]) + ' and ' + aspects[-1] if len(aspects) > 1 else aspects[0]} {'were' if len(aspects) > 1 else 'was'} excellent."
                
        elif sentiment_type == "negative":
            if opinions:
                return f"The {', '.join(aspects[:-1]) + ' and ' + aspects[-1] if len(aspects) > 1 else aspects[0]} {'were' if len(aspects) > 1 else 'was'} disappointing, mainly due to the {', '.join(opinions[:-1]) + ' and ' + opinions[-1] if len(opinions) > 1 else opinions[0]}."
            else:
                return f"The {', '.join(aspects[:-1]) + ' and ' + aspects[-1] if len(aspects) > 1 else aspects[0]} {'were' if len(aspects) > 1 else 'was'} disappointing."
                
        else:  # neutral
            if opinions:
                return f"The {', '.join(aspects[:-1]) + ' and ' + aspects[-1] if len(aspects) > 1 else aspects[0]} {'were' if len(aspects) > 1 else 'was'} average, with {', '.join(opinions[:-1]) + ' and ' + opinions[-1] if len(opinions) > 1 else opinions[0]}."
            else:
                return f"The {', '.join(aspects[:-1]) + ' and ' + aspects[-1] if len(aspects) > 1 else aspects[0]} {'were' if len(aspects) > 1 else 'was'} neither particularly good nor bad."