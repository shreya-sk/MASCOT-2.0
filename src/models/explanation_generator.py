# src/models/explanation_generator.py
import random

class ExplanationGenerator:
    """
    Improved template-based explanation generator for ABSA triplets
    
    This class creates natural language explanations from extracted triplets
    using a variety of templates and patterns for more natural-sounding explanations.
    """
    def __init__(self):
        # IMPROVED: More natural-sounding templates for different sentiment types
        self.positive_templates = [
            "The {aspect} is excellent because of its {opinion}.",
            "I liked the {aspect} due to its {opinion}.",
            "The {aspect} was great, especially the {opinion}.",
            "The {opinion} of the {aspect} was really good.",
            "Thanks to its {opinion}, the {aspect} was quite enjoyable.",
            "The {aspect} stands out with its {opinion}.",
            "One highlight was the {aspect} with its {opinion}.",
            "The {aspect} impressed me with its {opinion}.",
            "I appreciated the {opinion} of the {aspect}.",
            "The {aspect} deserves praise for its {opinion}."
        ]
        
        self.negative_templates = [
            "The {aspect} is disappointing because of its {opinion}.",
            "I didn't like the {aspect} due to its {opinion}.",
            "The {aspect} was below par, especially the {opinion}.",
            "The {opinion} of the {aspect} was really poor.",
            "Because of its {opinion}, the {aspect} was not enjoyable.",
            "The {aspect} was let down by its {opinion}.",
            "A disappointment was the {aspect} with its {opinion}.",
            "The {aspect} frustrated me with its {opinion}.",
            "I was dissatisfied with the {opinion} of the {aspect}.",
            "The {aspect} needs improvement in its {opinion}."
        ]
        
        self.neutral_templates = [
            "The {aspect} is average with {opinion}.",
            "The {aspect} was neither good nor bad with its {opinion}.",
            "The {opinion} of the {aspect} was just okay.",
            "The {aspect} had standard {opinion}.",
            "The {aspect} with its {opinion} was typical, nothing special.",
            "The {aspect} comes with {opinion}, which is acceptable.",
            "The {opinion} of the {aspect} meets basic expectations.",
            "The {aspect} has ordinary {opinion}, nothing remarkable.",
            "I found the {aspect} with its {opinion} to be adequate.",
            "The {aspect} and its {opinion} were middle-of-the-road."
        ]
        
        # Templates for empty opinions
        self.aspect_only_templates = {
            'POS': [
                "The {aspect} was excellent.",
                "I really enjoyed the {aspect}.",
                "The {aspect} was one of the highlights.",
                "The {aspect} was impressive.",
                "I was very pleased with the {aspect}.",
                "The {aspect} exceeded my expectations.",
                "The {aspect} was top-notch.",
                "The {aspect} was definitely a high point."
            ],
            'NEG': [
                "The {aspect} was disappointing.",
                "I didn't like the {aspect}.",
                "The {aspect} was a letdown.",
                "The {aspect} didn't meet expectations.",
                "I was dissatisfied with the {aspect}.",
                "The {aspect} needs improvement.",
                "The {aspect} was below standards.",
                "The {aspect} left much to be desired."
            ],
            'NEU': [
                "The {aspect} was average.",
                "The {aspect} was neither good nor bad.",
                "The {aspect} was just okay.",
                "The {aspect} was adequate.",
                "The {aspect} was unremarkable.",
                "The {aspect} was standard fare.",
                "The {aspect} met basic expectations.",
                "The {aspect} was middle-of-the-road."
            ]
        }
        
        # Templates for combining multiple triplets (same aspect, different opinions)
        self.multi_opinion_templates = [
            "The {aspect} was {sentiment} with {opinion_list}.",
            "I found the {aspect} to be {sentiment} because of {opinion_list}.",
            "The {aspect} {sentiment_verb} {opinion_list}.",
            "What made the {aspect} {sentiment} was {opinion_list}."
        ]
        
        # Templates for combining multiple triplets (different aspects, different sentiments)
        self.combo_templates = [
            "While the {pos_aspect} was excellent with its {pos_opinion}, the {neg_aspect} was disappointing due to {neg_opinion}.",
            "I enjoyed the {pos_aspect} because of its {pos_opinion}, but the {neg_aspect} could be improved as it was {neg_opinion}.",
            "The {pos_aspect} had great {pos_opinion}, though the {neg_aspect} was lacking with its {neg_opinion}.",
            "Despite the excellent {pos_opinion} of the {pos_aspect}, the {neg_aspect} was let down by its {neg_opinion}.",
            "The {pos_aspect} impressed me with its {pos_opinion}, while the {neg_aspect} disappointed with its {neg_opinion}.",
            "I liked the {pos_aspect}'s {pos_opinion}, but wished the {neg_aspect} wasn't so {neg_opinion}.",
            "The highlight was the {pos_aspect} with its {pos_opinion}, although the {neg_aspect}'s {neg_opinion} was a letdown."
        ]
        
        # Sentiment verbs for more natural phrasing
        self.sentiment_verbs = {
            'POS': ['impressed with', 'stood out for', 'excelled with', 'was enhanced by', 'shined with'],
            'NEG': ['suffered from', 'was undermined by', 'was hindered by', 'was compromised by', 'disappointed with'],
            'NEU': ['came with', 'featured', 'included', 'offered', 'provided']
        }
        
        # Sentiment adjectives for more natural phrasing
        self.sentiment_adjectives = {
            'POS': ['excellent', 'outstanding', 'impressive', 'great', 'superb', 'delightful'],
            'NEG': ['disappointing', 'poor', 'subpar', 'problematic', 'unsatisfactory', 'lacking'],
            'NEU': ['adequate', 'acceptable', 'standard', 'average', 'ordinary', 'typical']
        }
    
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
        
        # Group triplets by aspect
        aspect_groups = {}
        for triplet in triplets:
            aspect = triplet.get('aspect', '').strip()
            if not aspect:
                aspect = "general"
                
            if aspect not in aspect_groups:
                aspect_groups[aspect] = []
            aspect_groups[aspect].append(triplet)
        
        # Group triplets by sentiment
        positive_triplets = [t for t in triplets if t['sentiment'] == 'POS']
        negative_triplets = [t for t in triplets if t['sentiment'] == 'NEG']
        neutral_triplets = [t for t in triplets if t['sentiment'] == 'NEU']
        
        # If we have multiple aspects, generate an aspect-focused explanation
        if len(aspect_groups) > 1:
            return self._generate_multi_aspect_explanation(aspect_groups)
        
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
    
    def _generate_multi_aspect_explanation(self, aspect_groups):
        """Generate explanation focusing on multiple aspects"""
        explanations = []
        
        # Process each aspect group
        for aspect, triplets in aspect_groups.items():
            # Skip if aspect is empty
            if not aspect:
                continue
                
            # Get the majority sentiment for this aspect
            sentiments = [t.get('sentiment', 'NEU') for t in triplets]
            if not sentiments:
                continue
                
            sentiment_counts = {'POS': 0, 'NEG': 0, 'NEU': 0}
            for s in sentiments:
                sentiment_counts[s] += 1
                
            majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            # Collect all opinions for this aspect
            opinions = [t.get('opinion', '').strip() for t in triplets]
            opinions = [o for o in opinions if o]  # Remove empty opinions
            
            # If we have opinions, create a detailed explanation
            if opinions:
                # Deduplicate opinions
                unique_opinions = list(dict.fromkeys(opinions))
                
                if len(unique_opinions) == 1:
                    # Just one opinion
                    explanation = self._generate_single_explanation({
                        'aspect': aspect,
                        'opinion': unique_opinions[0],
                        'sentiment': majority_sentiment
                    })
                    explanations.append(explanation)
                else:
                    # Multiple opinions
                    if len(unique_opinions) == 2:
                        opinion_text = f"{unique_opinions[0]} and {unique_opinions[1]}"
                    else:
                        opinion_text = ", ".join(unique_opinions[:-1]) + f", and {unique_opinions[-1]}"
                    
                    # Get sentiment-appropriate phrasing
                    sentiment_adj = random.choice(self.sentiment_adjectives[majority_sentiment])
                    sentiment_verb = random.choice(self.sentiment_verbs[majority_sentiment])
                    
                    # Select a multi-opinion template
                    template = random.choice(self.multi_opinion_templates)
                    explanation = template.format(
                        aspect=aspect,
                        sentiment=sentiment_adj,
                        sentiment_verb=sentiment_verb,
                        opinion_list=opinion_text
                    )
                    explanations.append(explanation)
            else:
                # No opinions, use aspect-only template
                templates = self.aspect_only_templates.get(majority_sentiment, self.aspect_only_templates['NEU'])
                template = random.choice(templates)
                explanation = template.format(aspect=aspect)
                explanations.append(explanation)
        
        # Combine all explanations
        if explanations:
            return " ".join(explanations)
        else:
            return "The review mentioned several aspects but didn't provide specific details."
    
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
            return f"The {pos_aspect} was mixed - the {pos_opinion} was excellent, but the {neg_opinion} was disappointing."
        
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
                if len(aspects) == 1:
                    return f"The {aspects[0]} was excellent, with {self._join_items(opinions)}."
                else:
                    return f"Several aspects were excellent: {self._join_items(aspects)}, with {self._join_items(opinions)}."
            else:
                if len(aspects) == 1:
                    return f"The {aspects[0]} was excellent."
                else:
                    return f"Several aspects were excellent: {self._join_items(aspects)}."
                
        elif sentiment_type == "negative":
            if opinions:
                if len(aspects) == 1:
                    return f"The {aspects[0]} was disappointing, due to {self._join_items(opinions)}."
                else:
                    return f"Several aspects were disappointing: {self._join_items(aspects)}, with issues like {self._join_items(opinions)}."
            else:
                if len(aspects) == 1:
                    return f"The {aspects[0]} was disappointing."
                else:
                    return f"Several aspects were disappointing: {self._join_items(aspects)}."
                
        else:  # neutral
            if opinions:
                if len(aspects) == 1:
                    return f"The {aspects[0]} was average, with {self._join_items(opinions)}."
                else:
                    return f"Several aspects were average: {self._join_items(aspects)}, with {self._join_items(opinions)}."
            else:
                if len(aspects) == 1:
                    return f"The {aspects[0]} was neither particularly good nor bad."
                else:
                    return f"Several aspects were average: {self._join_items(aspects)}."
    
    def _join_items(self, items):
        """Helper function to join items in a grammatically correct way"""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"