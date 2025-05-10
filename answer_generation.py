import json
import os
import faiss
import pickle
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm.auto import tqdm
from typing import List, Dict, Any
import argparse
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MedicalQAEvaluator:
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-base",
        test_data_path: str = None,
        faiss_index_path: str = None,
        mapping_file_path: str = None,
        batch_size: int = 8,
        top_k: int = 4,
        max_input_length: int = 512,
        max_output_length: int = 256,  # Increased from 128 to allow for longer answers
        use_auth_token: bool = False
    ):
        """
        Initialize the Medical QA Evaluator.
        
        Args:
            t5_model_name: HuggingFace model name/ID (e.g., 'google/flan-t5-base', 'google/flan-t5-large', etc.)
            test_data_path: Path to test data JSON file
            faiss_index_path: Path to FAISS index file
            mapping_file_path: Path to index mapping pickle file
            batch_size: Batch size for processing
            top_k: Number of similar QA pairs to retrieve
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            use_auth_token: Whether to use HuggingFace auth token for private models
        """
        self.batch_size = batch_size
        self.top_k = top_k
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_auth_token = use_auth_token
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models and data
        self._load_t5_model(t5_model_name)
        self._load_biobert_model()
        
        if test_data_path and faiss_index_path and mapping_file_path:
            self._load_test_data(test_data_path)
            self._load_faiss_index(faiss_index_path, mapping_file_path)
        else:
            logger.warning("Test data, FAISS index, or mapping paths not provided. Load them manually before evaluation.")
        
        # BLEU smoothing function
        self.smoother = SmoothingFunction().method1
        
    def _load_t5_model(self, model_name: str):
        """
        Load the T5 model and tokenizer directly from HuggingFace.
        
        Args:
            model_name: HuggingFace model ID (e.g., 'google/flan-t5-base')
        """
        try:
            logger.info(f"Loading T5 model from HuggingFace: {model_name}")
            
            # Load tokenizer from HuggingFace
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                use_auth_token=self.use_auth_token
            )
            
            # Load model from HuggingFace
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                use_auth_token=self.use_auth_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use fp16 if on GPU
            ).to(self.device)
            
            logger.info(f"Successfully loaded {model_name} from HuggingFace")
            
        except Exception as e:
            logger.error(f"Error loading T5 model from HuggingFace: {e}")
            raise
            
    def _load_biobert_model(self):
        """Load the BioBERT model for embeddings."""
        try:
            logger.info("Loading BioBERT model")
            self.biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(self.device)
            self.biobert_model.eval()  # Set to evaluation mode
            logger.info("BioBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BioBERT model: {e}")
            raise
            
    def _load_test_data(self, test_data_path: str):
        """Load the test dataset."""
        try:
            logger.info(f"Loading test data from {test_data_path}")
            with open(test_data_path, 'r', encoding='utf-8') as f:
                self.test_data = json.load(f)
            logger.info(f"Loaded {len(self.test_data)} test examples")
        except FileNotFoundError:
            logger.error(f"Test data file not found: {test_data_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in test data file: {test_data_path}")
            raise
            
    def _load_faiss_index(self, index_path: str, mapping_path: str):
        """Load the FAISS index and mapping file."""
        try:
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS index contains {self.index.ntotal} vectors")
            
            logger.info(f"Loading index mapping from {mapping_path}")
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
            
            # Ensure all keys are strings for consistent lookup
            if isinstance(next(iter(mapping['index_to_qa'].keys()), 0), int):
                self.index_to_qa = {str(k): v for k, v in mapping['index_to_qa'].items()}
                logger.info("Converted integer keys to strings in index mapping")
            else:
                self.index_to_qa = mapping['index_to_qa']
                
            logger.info(f"Loaded mapping with {len(self.index_to_qa)} entries")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading FAISS index or mapping: {e}")
            raise
            
    def embed_questions(self, questions: List[str]) -> np.ndarray:
        """
        Generate BioBERT embeddings for a batch of questions.
        
        Args:
            questions: List of question strings
            
        Returns:
            Numpy array of embeddings
        """
        # Tokenize all questions at once
        inputs = self.biobert_tokenizer(
            questions, 
            return_tensors="pt", 
            max_length=self.max_input_length, 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        # Generate embeddings
        embeddings = []
        with torch.no_grad():
            outputs = self.biobert_model(**inputs)
            
            # Apply mean pooling to each question separately
            for i in range(len(questions)):
                # Extract single question tensors
                question_tokens = outputs.last_hidden_state[i].unsqueeze(0)
                question_mask = inputs['attention_mask'][i].unsqueeze(0).unsqueeze(-1)
                
                # Mean pooling
                embedding = torch.sum(question_tokens * question_mask, 1) / torch.sum(question_mask, 1)
                embeddings.append(embedding.cpu().numpy())
                
        # Stack and normalize
        embeddings = np.vstack(embeddings)
        faiss.normalize_L2(embeddings)
        return embeddings
    
    def retrieve_similar_qa_pairs(self, questions: List[str]) -> List[List[Dict]]:
        """
        Retrieve top-k similar QA pairs for each question.
        
        Args:
            questions: List of question strings
            
        Returns:
            List of lists of retrieved QA pairs
        """
        # Generate embeddings for all questions
        query_embeddings = self.embed_questions(questions)
        
        # Search the FAISS index
        D, I = self.index.search(query_embeddings, self.top_k + 1)  # +1 to handle self-matches
        
        # Extract the QA pairs for each question
        all_retrieved_qa = []
        for i, indices in enumerate(I):
            retrieved_qa = []
            question = questions[i]
            
            for idx in indices:
                idx_key = str(idx)
                
                # Handle potential key errors
                if idx_key not in self.index_to_qa:
                    logger.warning(f"Index {idx} not found in mapping. Skipping.")
                    continue
                    
                qa_pair = self.index_to_qa[idx_key]
                
                # Skip if the retrieved question is identical to the query
                if qa_pair['question'].strip() == question.strip():
                    continue
                    
                retrieved_qa.append(qa_pair)
                
                # Only keep top_k pairs
                if len(retrieved_qa) >= self.top_k:
                    break
                    
            all_retrieved_qa.append(retrieved_qa)
            
        return all_retrieved_qa
    
    def format_inputs(self, questions: List[str], all_retrieved_qa: List[List[Dict]]) -> List[str]:
        """
        Format inputs for the T5 model.
        
        Args:
            questions: List of question strings
            all_retrieved_qa: List of lists of retrieved QA pairs
            
        Returns:
            List of formatted input strings
        """
        formatted_inputs = []
        
        for question, retrieved_qa in zip(questions, all_retrieved_qa):
            # Create context string
            context_parts = []
            for qa in retrieved_qa:
                context_parts.append(f"Question: {qa['question']} Answer: {qa['answer']}")
            
            context = " ".join(context_parts)
            # Role-based prompt to encourage medical expert-like answers
            input_text = f"Act as a Clinical Medical Expert. Using the following context information, provide a detailed, authoritative answer to the medical question in 3-4 sentences with evidence-based explanations. Context: {context} Question: {question} Expert Answer:"
            formatted_inputs.append(input_text)
            
        return formatted_inputs
    
    def generate_answers(self, formatted_inputs: List[str]) -> List[str]:
        """
        Generate answers using the T5 model.
        
        Args:
            formatted_inputs: List of formatted input strings
            
        Returns:
            List of generated answer strings
        """
        generated_answers = []
        
        # Process in batches to avoid OOM errors
        for i in range(0, len(formatted_inputs), self.batch_size):
            batch_inputs = formatted_inputs[i:i+self.batch_size]
            
            # Tokenize inputs
            inputs = self.t5_tokenizer(
                batch_inputs,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate outputs with modified parameters for longer, more detailed answers
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    min_length=50,  # Encourage longer answers
                    num_beams=5,    # Increase beam search for better quality
                    length_penalty=1.5,  # Encourage longer generations
                    early_stopping=False,  # Don't stop early
                    no_repeat_ngram_size=3,
                    do_sample=True,  # Use sampling for more diverse answers
                    top_p=0.9,       # Use nucleus sampling
                    temperature=0.8  # Control randomness
                )
            
            # Decode outputs
            batch_answers = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_answers.extend(batch_answers)
            
        return generated_answers
    
    def calculate_bleu(self, generated_answers: List[str], reference_answers: List[str]) -> Dict[str, Any]:
        """
        Calculate BLEU scores for generated answers.
        
        Args:
            generated_answers: List of generated answer strings
            reference_answers: List of reference answer strings
            
        Returns:
            Dictionary with BLEU scores
        """
        bleu_scores = []
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            # Tokenize using NLTK's word_tokenize for better handling
            reference = [nltk.word_tokenize(ref.lower())]
            candidate = nltk.word_tokenize(gen.lower())
            
            # Calculate BLEU scores with smoothing
            try:
                # Calculate individual n-gram scores
                bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=self.smoother)
                bleu_2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=self.smoother)
                bleu_3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=self.smoother)
                bleu_4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=self.smoother)
                
                # Calculate the cumulative BLEU score
                bleu = sentence_bleu(reference, candidate, smoothing_function=self.smoother)
                
                bleu_1_scores.append(bleu_1)
                bleu_2_scores.append(bleu_2)
                bleu_3_scores.append(bleu_3)
                bleu_4_scores.append(bleu_4)
                bleu_scores.append(bleu)
            except Exception as e:
                logger.warning(f"Error calculating BLEU score: {e}")
                bleu_1_scores.append(0)
                bleu_2_scores.append(0)
                bleu_3_scores.append(0)
                bleu_4_scores.append(0)
                bleu_scores.append(0)
        
        # Calculate statistics
        results = {
            'bleu_scores': bleu_scores,
            'avg_bleu': float(np.mean(bleu_scores)),
            'median_bleu': float(np.median(bleu_scores)),
            'std_bleu': float(np.std(bleu_scores)),
            'bleu_1': float(np.mean(bleu_1_scores)),
            'bleu_2': float(np.mean(bleu_2_scores)),
            'bleu_3': float(np.mean(bleu_3_scores)),
            'bleu_4': float(np.mean(bleu_4_scores))
        }
        
        return results
    
    def evaluate(self, output_file: str = 'evaluation_results.json', sample_limit: int = 5000):
        """
        Run the full evaluation pipeline.
        
        Args:
            output_file: Path to save the evaluation results
            sample_limit: Maximum number of test samples to evaluate
        """
        logger.info("Starting evaluation")
        
        # Limit to sample_limit test examples
        test_sample = self.test_data[:sample_limit]
        logger.info(f"Limited evaluation to {len(test_sample)} samples out of {len(self.test_data)} total")
        
        questions = [ex['question'] for ex in test_sample]
        reference_answers = [ex['answer'] for ex in test_sample]
        
        results = defaultdict(list)
        generated_answers = []
        
        # Process in batches to manage memory
        batch_size = self.batch_size * 2  # Can use larger batch for retrieval
        
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating answers"):
            batch_questions = questions[i:i+batch_size]
            batch_ref_answers = reference_answers[i:i+batch_size]
            
            # Retrieve similar QA pairs
            batch_retrieved_qa = self.retrieve_similar_qa_pairs(batch_questions)
            
            # Print retrieved queries for a sample of questions (just for the first batch)
            if i == 0:
                num_samples = min(3, len(batch_questions))
                for j in range(num_samples):
                    logger.info(f"\n{'='*50}")
                    logger.info(f"ORIGINAL QUESTION: {batch_questions[j]}")
                    logger.info(f"{'='*50}")
                    logger.info("RETRIEVED QA PAIRS:")
                    for idx, qa_pair in enumerate(batch_retrieved_qa[j]):
                        logger.info(f"\n--- Pair {idx+1} ---")
                        logger.info(f"Question: {qa_pair['question']}")
                        logger.info(f"Answer: {qa_pair['answer']}")
                    logger.info(f"{'='*50}\n")
            
            # Format inputs
            batch_formatted_inputs = self.format_inputs(batch_questions, batch_retrieved_qa)
            
            # Generate answers
            batch_answers = self.generate_answers(batch_formatted_inputs)
            generated_answers.extend(batch_answers)
            
            # Print answers for each question in this batch
            for j in range(len(batch_questions)):
                logger.info(f"\n{'='*80}")
                logger.info(f"QUESTION: {batch_questions[j]}")
                logger.info(f"GENERATED ANSWER: {batch_answers[j]}")
                logger.info(f"REFERENCE ANSWER: {batch_ref_answers[j]}")
                logger.info(f"{'='*80}\n")
        
        # Calculate BLEU scores
        logger.info("Calculating BLEU scores")
        bleu_results = self.calculate_bleu(generated_answers, reference_answers)
        
        # Prepare full results
        for idx, (gen, ref) in enumerate(zip(generated_answers, reference_answers)):
            results['question'].append(questions[idx])
            results['generated_answer'].append(gen)
            results['reference_answer'].append(ref)
            results['bleu_score'].append(bleu_results['bleu_scores'][idx])
        
        # Add summary statistics
        results['summary'] = {
            'avg_bleu': bleu_results['avg_bleu'],
            'median_bleu': bleu_results['median_bleu'],
            'std_bleu': bleu_results['std_bleu'],
            'bleu_1': bleu_results['bleu_1'],
            'bleu_2': bleu_results['bleu_2'],
            'bleu_3': bleu_results['bleu_3'],
            'bleu_4': bleu_results['bleu_4']
        }
        
        # Log summary results
        logger.info(f"Evaluation Summary:")
        logger.info(f"Average BLEU: {bleu_results['avg_bleu']:.4f}")
        logger.info(f"BLEU-1: {bleu_results['bleu_1']:.4f}")
        logger.info(f"BLEU-2: {bleu_results['bleu_2']:.4f}")
        logger.info(f"BLEU-3: {bleu_results['bleu_3']:.4f}")
        logger.info(f"BLEU-4: {bleu_results['bleu_4']:.4f}")
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate T5 model on medical QA dataset")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base", 
                        help="HuggingFace T5 model name (e.g., google/flan-t5-base, google/flan-t5-large)")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Path to test data JSON file")
    parser.add_argument("--faiss_index", type=str, required=True, 
                        help="Path to FAISS index file")
    parser.add_argument("--mapping_file", type=str, required=True, 
                        help="Path to index mapping pickle file")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", 
                        help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing")
    parser.add_argument("--top_k", type=int, default=4, 
                        help="Number of similar QA pairs to retrieve")
    parser.add_argument("--use_auth_token", action="store_true",
                        help="Use HuggingFace auth token for private models")
    parser.add_argument("--sample_limit", type=int, default=5000,
                        help="Maximum number of test samples to evaluate")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MedicalQAEvaluator(
        t5_model_name=args.model_name,
        test_data_path=args.test_data,
        faiss_index_path=args.faiss_index,
        mapping_file_path=args.mapping_file,
        batch_size=args.batch_size,
        top_k=args.top_k,
        use_auth_token=args.use_auth_token
    )
    
    # Run evaluation with sample limit
    evaluator.evaluate(args.output_file, args.sample_limit)

if __name__ == "__main__":
    main()