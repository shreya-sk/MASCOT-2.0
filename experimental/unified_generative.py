# src/models/unified_generative.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Any, Optional

class UnifiedGenerativeABSA(nn.Module):
    """
    Unified generative framework converting all ABSA subtasks to sequence generation
    Based on InstructABSA paradigm achieving 9.59% performance gains
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use T5 as base model (following InstructABSA success)
        model_name = getattr(config, 'generative_model', 't5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Task-specific instruction templates
        self.instruction_templates = {
            'aspect_extraction': "Extract all aspect terms from the following text: {text}",
            'opinion_extraction': "Extract all opinion terms from the following text: {text}",
            'sentiment_classification': "Classify the sentiment of aspect '{aspect}' in: {text}",
            'triplet_extraction': "Extract all (aspect, opinion, sentiment) triplets from: {text}",
            'quadruple_extraction': "Extract all (aspect, category, opinion, sentiment) quadruples from: {text}",
            'implicit_detection': "Identify implicit aspects and opinions in: {text}"
        }
        
        # Output format templates
        self.output_templates = {
            'aspect_extraction': "aspects: {aspects}",
            'opinion_extraction': "opinions: {opinions}",
            'sentiment_classification': "sentiment: {sentiment}",
            'triplet_extraction': "triplets: {triplets}",
            'quadruple_extraction': "quadruples: {quadruples}",
            'implicit_detection': "implicit: {implicit_elements}"
        }
        
        # Special tokens for structured output
        special_tokens = [
            "<aspect>", "</aspect>", "<opinion>", "</opinion>", 
            "<sentiment>", "</sentiment>", "<category>", "</category>",
            "<triplet>", "</triplet>", "<quadruple>", "</quadruple>",
            "<implicit>", "</implicit>", "<POS>", "<NEG>", "<NEU>"
        ]
        
        # Add special tokens to tokenizer
        self.tokenizer.add_tokens(special_tokens)
        self.t5_model.resize_token_embeddings(len(self.tokenizer))
        
        # Task classifier for multi-task learning
        self.task_classifier = nn.Linear(self.t5_model.config.d_model, len(self.instruction_templates))
        
        # Contrastive learning components
        self.contrastive_projector = nn.Sequential(
            nn.Linear(self.t5_model.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def prepare_instruction_input(self, text: str, task: str, aspect: str = None) -> str:
        """Prepare instruction-formatted input"""
        template = self.instruction_templates[task]
        
        if aspect and '{aspect}' in template:
            instruction = template.format(text=text, aspect=aspect)
        else:
            instruction = template.format(text=text)
            
        return instruction
    
    def prepare_structured_output(self, task: str, elements: List[Any]) -> str:
        """Prepare structured output format"""
        if task == 'aspect_extraction':
            aspects = ", ".join([f"<aspect>{asp}</aspect>" for asp in elements])
            return f"aspects: {aspects}"
            
        elif task == 'opinion_extraction':
            opinions = ", ".join([f"<opinion>{op}</opinion>" for op in elements])
            return f"opinions: {opinions}"
            
        elif task == 'sentiment_classification':
            return f"sentiment: <sentiment>{elements}</sentiment>"
            
        elif task == 'triplet_extraction':
            triplets = []
            for asp, op, sent in elements:
                triplet = f"<triplet><aspect>{asp}</aspect><opinion>{op}</opinion><sentiment>{sent}</sentiment></triplet>"
                triplets.append(triplet)
            return f"triplets: {', '.join(triplets)}"
            
        elif task == 'quadruple_extraction':
            quadruples = []
            for asp, cat, op, sent in elements:
                quad = f"<quadruple><aspect>{asp}</aspect><category>{cat}</category><opinion>{op}</opinion><sentiment>{sent}</sentiment></quadruple>"
                quadruples.append(quad)
            return f"quadruples: {', '.join(quadruples)}"
            
        elif task == 'implicit_detection':
            implicit_elements = ", ".join([f"<implicit>{elem}</implicit>" for elem in elements])
            return f"implicit: {implicit_elements}"
            
        return str(elements)
    
    def forward(self, input_texts: List[str], target_outputs: List[str] = None, 
                task: str = 'triplet_extraction', training: bool = True):
        """
        Unified forward pass for all ABSA tasks
        
        Args:
            input_texts: List of input texts
            target_outputs: List of target structured outputs (for training)
            task: ABSA task type
            training: Whether in training mode
        """
        batch_size = len(input_texts)
        device = next(self.parameters()).device
        
        # Prepare instruction inputs
        instruction_inputs = [
            self.prepare_instruction_input(text, task) for text in input_texts
        ]
        
        # Tokenize inputs
        input_encodings = self.tokenizer(
            instruction_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        if training and target_outputs:
            # Tokenize targets for training
            target_encodings = self.tokenizer(
                target_outputs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass with targets
            outputs = self.t5_model(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                labels=target_encodings.input_ids
            )
            
            # Get encoder hidden states for contrastive learning
            encoder_outputs = self.t5_model.encoder(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask
            )
            
            # Pool encoder outputs
            pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)  # [batch_size, d_model]
            
            # Task classification
            task_logits = self.task_classifier(pooled_output)
            
            # Contrastive representations
            contrastive_repr = self.contrastive_projector(pooled_output)
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'task_logits': task_logits,
                'contrastive_repr': contrastive_repr,
                'encoder_hidden_states': encoder_outputs.last_hidden_state
            }
        else:
            # Inference mode
            generated_ids = self.t5_model.generate(
                input_ids=input_encodings.input_ids,
                attention_mask=input_encodings.attention_mask,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
            
            # Decode generated outputs
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            
            return {
                'generated_texts': generated_texts,
                'generated_ids': generated_ids
            }
    
    def parse_generated_output(self, generated_text: str, task: str) -> List[Any]:
        """Parse structured generated output back to elements"""
        if task == 'aspect_extraction':
            return self._parse_aspects(generated_text)
        elif task == 'opinion_extraction':
            return self._parse_opinions(generated_text)
        elif task == 'sentiment_classification':
            return self._parse_sentiment(generated_text)
        elif task == 'triplet_extraction':
            return self._parse_triplets(generated_text)
        elif task == 'quadruple_extraction':
            return self._parse_quadruples(generated_text)
        elif task == 'implicit_detection':
            return self._parse_implicit(generated_text)
        
        return []
    
    def _parse_aspects(self, text: str) -> List[str]:
        """Parse aspect terms from generated text"""
        import re
        aspects = re.findall(r'<aspect>(.*?)</aspect>', text)
        return [asp.strip() for asp in aspects]
    
    def _parse_opinions(self, text: str) -> List[str]:
        """Parse opinion terms from generated text"""
        import re
        opinions = re.findall(r'<opinion>(.*?)</opinion>', text)
        return [op.strip() for op in opinions]
    
    def _parse_sentiment(self, text: str) -> str:
        """Parse sentiment from generated text"""
        import re
        sentiment_match = re.search(r'<sentiment>(.*?)</sentiment>', text)
        return sentiment_match.group(1).strip() if sentiment_match else 'NEU'
    
    def _parse_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Parse triplets from generated text"""
        import re
        triplets = []
        
        triplet_matches = re.findall(r'<triplet>(.*?)</triplet>', text)
        for triplet_text in triplet_matches:
            aspect_match = re.search(r'<aspect>(.*?)</aspect>', triplet_text)
            opinion_match = re.search(r'<opinion>(.*?)</opinion>', triplet_text)
            sentiment_match = re.search(r'<sentiment>(.*?)</sentiment>', triplet_text)
            
            if aspect_match and opinion_match and sentiment_match:
                triplets.append((
                    aspect_match.group(1).strip(),
                    opinion_match.group(1).strip(),
                    sentiment_match.group(1).strip()
                ))
        
        return triplets
    
    def _parse_quadruples(self, text: str) -> List[Tuple[str, str, str, str]]:
        """Parse quadruples from generated text"""
        import re
        quadruples = []
        
        quad_matches = re.findall(r'<quadruple>(.*?)</quadruple>', text)
        for quad_text in quad_matches:
            aspect_match = re.search(r'<aspect>(.*?)</aspect>', quad_text)
            category_match = re.search(r'<category>(.*?)</category>', quad_text)
            opinion_match = re.search(r'<opinion>(.*?)</opinion>', quad_text)
            sentiment_match = re.search(r'<sentiment>(.*?)</sentiment>', quad_text)
            
            if all([aspect_match, category_match, opinion_match, sentiment_match]):
                quadruples.append((
                    aspect_match.group(1).strip(),
                    category_match.group(1).strip(),
                    opinion_match.group(1).strip(),
                    sentiment_match.group(1).strip()
                ))
        
        return quadruples
    
    def _parse_implicit(self, text: str) -> List[str]:
        """Parse implicit elements from generated text"""
        import re
        implicit_elements = re.findall(r'<implicit>(.*?)</implicit>', text)
        return [elem.strip() for elem in implicit_elements]

class MultiTaskGenerativeLoss(nn.Module):
    """
    Specialized loss function for unified generative ABSA
    """
    
    def __init__(self, config):
        super().__init__()
        self.generation_weight = getattr(config, 'generation_loss_weight', 1.0)
        self.task_weight = getattr(config, 'task_classification_weight', 0.5)
        self.contrastive_weight = getattr(config, 'contrastive_loss_weight', 0.3)
        
        # Task classification loss
        self.task_criterion = nn.CrossEntropyLoss()
        
        # Contrastive loss
        self.contrastive_criterion = nn.CrossEntropyLoss()
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        
    def forward(self, outputs, task_labels=None, contrastive_labels=None):
        """
        Compute multi-task generative loss
        
        Args:
            outputs: Model outputs dictionary
            task_labels: Task classification labels
            contrastive_labels: Labels for contrastive learning
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Generation loss (from T5)
        if 'loss' in outputs:
            generation_loss = outputs['loss']
            total_loss += self.generation_weight * generation_loss
            loss_dict['generation_loss'] = generation_loss.item()
        
        # Task classification loss
        if 'task_logits' in outputs and task_labels is not None:
            task_loss = self.task_criterion(outputs['task_logits'], task_labels)
            total_loss += self.task_weight * task_loss
            loss_dict['task_loss'] = task_loss.item()
        
        # Contrastive loss
        if 'contrastive_repr' in outputs and contrastive_labels is not None:
            contrastive_loss = self._compute_contrastive_loss(
                outputs['contrastive_repr'], contrastive_labels
            )
            total_loss += self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_contrastive_loss(self, representations, labels):
        """Compute supervised contrastive loss"""
        batch_size = representations.size(0)
        
        # Normalize representations
        representations = F.normalize(representations, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create positive mask (same labels)
        labels = labels.unsqueeze(0)
        positive_mask = torch.eq(labels, labels.T).float()
        positive_mask.fill_diagonal_(0)  # Remove self-similarity
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Sum of all similarities (denominator)
        sum_exp = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # Positive similarities
        positive_sim = exp_sim * positive_mask
        
        # Compute loss for each sample
        losses = []
        for i in range(batch_size):
            if positive_mask[i].sum() > 0:  # If there are positive pairs
                pos_sum = torch.sum(positive_sim[i])
                loss = -torch.log(pos_sum / sum_exp[i])
                losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=representations.device, requires_grad=True)