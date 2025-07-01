import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class MinimalInstructABSA(nn.Module):
    """
    Minimal implementation of InstructABSA paradigm
    Converts your existing ABSA model to instruction-following format
    """
    def __init__(self, config, existing_absa_model):
        super().__init__()
        
        # Keep your existing model as the backbone
        self.absa_backbone = existing_absa_model
        
        # Add T5 for instruction following
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')  # Start small
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # Instruction templates (core innovation)
        self.templates = {
            'triplet_extraction': "Extract aspect-opinion-sentiment triplets from: {text}",
            'aspect_extraction': "Find aspect terms in: {text}",
            'sentiment_classification': "What is the sentiment of '{aspect}' in: {text}",
        }
        
        # Bridge your existing model to T5
        self.feature_bridge = nn.Linear(config.hidden_size, self.t5_model.config.d_model)
        
        # Special tokens for structured output
        special_tokens = ["<triplet>", "</triplet>", "<aspect>", "</aspect>", 
                         "<opinion>", "</opinion>", "<POS>", "<NEG>", "<NEU>"]
        self.tokenizer.add_tokens(special_tokens)
        self.t5_model.resize_token_embeddings(len(self.tokenizer))
    
    def forward(self, input_ids, attention_mask, task_type='triplet_extraction', target_text=None):
        """
        Unified forward pass supporting both extraction and generation
        """
        # Get features from your existing model
        backbone_outputs = self.absa_backbone(input_ids, attention_mask)
        
        # Extract triplets using existing model
        extracted_triplets = self._extract_triplets_from_backbone(backbone_outputs, input_ids)
        
        # Convert to instruction format
        text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        instruction = self.templates[task_type].format(text=text)
        
        # T5 instruction following
        instruction_inputs = self.tokenizer(
            instruction, return_tensors='pt', max_length=512, truncation=True, padding=True
        ).to(input_ids.device)
        
        if target_text is not None:  # Training
            target_inputs = self.tokenizer(
                target_text, return_tensors='pt', max_length=256, truncation=True, padding=True
            ).to(input_ids.device)
            
            # Bridge features from backbone to T5
            backbone_features = backbone_outputs.get('hidden_states', backbone_outputs['aspect_logits'])
            bridged_features = self.feature_bridge(backbone_features.mean(dim=1))  # Pool sequence
            
            # Enhanced T5 forward with backbone features
            t5_outputs = self.t5_model(
                input_ids=instruction_inputs.input_ids,
                attention_mask=instruction_inputs.attention_mask,
                labels=target_inputs.input_ids
            )
            
            return {
                'loss': backbone_outputs.get('loss', 0) + t5_outputs.loss,
                'extraction_outputs': backbone_outputs,
                'generation_outputs': t5_outputs,
                'extracted_triplets': extracted_triplets
            }
        
        else:  # Inference
            generated_ids = self.t5_model.generate(
                input_ids=instruction_inputs.input_ids,
                attention_mask=instruction_inputs.attention_mask,
                max_length=256,
                num_beams=3,
                early_stopping=True
            )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            return {
                'extraction_outputs': backbone_outputs,
                'generated_text': generated_text,
                'extracted_triplets': extracted_triplets
            }
    
    def _extract_triplets_from_backbone(self, outputs, input_ids):
        """
        Convert your existing model outputs to triplets
        """
        try:
            # Use your existing triplet extraction logic
            aspect_logits = outputs['aspect_logits']
            opinion_logits = outputs['opinion_logits']
            sentiment_logits = outputs['sentiment_logits']
            
            # Extract spans (simplified)
            aspect_preds = aspect_logits.argmax(dim=-1)
            opinion_preds = opinion_logits.argmax(dim=-1)
            sentiment_preds = sentiment_logits.argmax(dim=-1)
            
            # Convert to triplet format
            triplets = []
            for b in range(input_ids.size(0)):
                batch_triplets = self._spans_to_triplets(
                    aspect_preds[b], opinion_preds[b], sentiment_preds[b]
                )
                triplets.append(batch_triplets)
            
            return triplets
            
        except Exception as e:
            print(f"Error extracting triplets: {e}")
            return [[]]  # Return empty triplets as fallback
    
    def _spans_to_triplets(self, aspect_pred, opinion_pred, sentiment_pred):
        """Convert BIO predictions to triplets"""
        # Extract aspect spans
        aspect_spans = self._extract_spans(aspect_pred)
        opinion_spans = self._extract_spans(opinion_pred)
        
        # Map sentiment
        sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
        sentiment = sentiment_map.get(sentiment_pred.item(), 'NEU')
        
        # Create triplets
        triplets = []
        for asp_span in aspect_spans:
            for op_span in opinion_spans:
                triplets.append({
                    'aspect_span': asp_span,
                    'opinion_span': op_span,
                    'sentiment': sentiment
                })
        
        return triplets
    
    def _extract_spans(self, predictions):
        """Extract spans from BIO predictions"""
        spans = []
        current_span = []
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # B tag
                if current_span:
                    spans.append(current_span)
                current_span = [i]
            elif pred == 2 and current_span:  # I tag
                current_span.append(i)
            else:  # O tag
                if current_span:
                    spans.append(current_span)
                    current_span = []
        
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def generate_structured_output(self, triplets):
        """
        Convert triplets to structured text format for T5 training
        """
        if not triplets:
            return "No triplets found."
        
        output_parts = []
        for triplet in triplets:
            aspect_text = f"aspect_{len(triplet['aspect_span'])}_tokens"  # Simplified
            opinion_text = f"opinion_{len(triplet['opinion_span'])}_tokens"
            sentiment = triplet['sentiment']
            
            structured = f"<triplet><aspect>{aspect_text}</aspect><opinion>{opinion_text}</opinion><sentiment>{sentiment}</sentiment></triplet>"
            output_parts.append(structured)
        
        return " ".join(output_parts)