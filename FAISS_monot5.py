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
from typing import List, Dict, Any, Tuple, Optional
import argparse
import logging
from collections import defaultdict
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import evaluate

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
        monot5_model_name: str = "castorini/monot5-base-msmarco",
        test_data_path: str = None,
        faiss_index_path: str = None,
        mapping_file_path: str = None,
        batch_size: int = 8,
        initial_k: int = 16,  # Retrieve more candidates for reranking
        top_k: int = 4,       # Final number of QA pairs after reranking
        max_input_length: int = 512,
        max_output_length: int = 256,  # Increased from 128 to allow for longer answers
        use_auth_token: bool = False
    ):
        """
        Initialize the Medical QA Evaluator with MonoT5 reranking.
        
        Args:
            t5_model_name: HuggingFace model name for the answer generation T5
            monot5_model_name: HuggingFace model name for MonoT5 reranker
            test_data_path: Path to test data JSON file
            faiss_index_path: Path to FAISS index file
            mapping_file_path: Path to index mapping pickle file
            batch_size: Batch size for processing
            initial_k: Number of QA pairs to initially retrieve (before reranking)
            top_k: Number of QA pairs to keep after reranking
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            use_auth_token: Whether to use HuggingFace auth token for private models
        """
        self.batch_size = batch_size
        self.initial_k = initial_k
        self.top_k = top_k
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_auth_token = use_auth_token
        self.monot5_model_name = monot5_model_name
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models and data
        self._load_t5_model(t5_model_name)
        self._load_biobert_model()
        self._load_monot5_model()
        
        if test_data_path and faiss_index_path and mapping_file_path:
            self._load_test_data(test_data_path)
            self._load_faiss_index(faiss_index_path, mapping_file_path)
        else:
            logger.warning("Test data, FAISS index, or mapping paths not provided. Load them manually before evaluation.")
        
        # By default, use all metrics
        self.use_bleu = True
        self.use_rouge = True
        self.use_bertscore = True
        self.use_meteor = True
        
        # BLEU smoothing function
        self.smoother = SmoothingFunction().method1
        
        # Check for required packages
        try:
            import rouge_score
            import bert_score
            import evaluate
            logger.info("All required evaluation packages are available")
        except ImportError as e:
            logger.warning(f"Missing evaluation package: {e}")
            logger.warning("Some evaluation metrics may not be available.")
            logger.warning("To install required packages: pip install rouge-score bert-score evaluate")
            
            # Disable metrics for missing packages
            if 'rouge_score' in str(e):
                self.use_rouge = False
            if 'bert_score' in str(e):
                self.use_bertscore = False
            if 'evaluate' in str(e):
                self.use_meteor = False
        
    def _load_t5_model(self, model_name: str):
        """Load the T5 model and tokenizer directly from HuggingFace."""
        logger.info(f"Loading T5 model from HuggingFace: {model_name}")
        
        # Load tokenizer and model with try/except
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            use_auth_token=self.use_auth_token
        )
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            use_auth_token=self.use_auth_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use fp16 if on GPU
        ).to(self.device)
        
        logger.info(f"Successfully loaded {model_name}")
            
    def _load_biobert_model(self):
        """Load the BioBERT model for embeddings."""
        logger.info("Loading BioBERT model")
        self.biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(self.device)
        self.biobert_model.eval()  # Set to evaluation mode
        logger.info("BioBERT model loaded successfully")
    
    def _load_monot5_model(self):
        """Load the MonoT5 model for reranking."""
        logger.info(f"Loading MonoT5 model: {self.monot5_model_name}")
        
        # Load tokenizer for MonoT5
        self.monot5_tokenizer = T5Tokenizer.from_pretrained(self.monot5_model_name)
        
        # Load the model
        self.monot5_model = T5ForConditionalGeneration.from_pretrained(
            self.monot5_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        self.monot5_model.eval()  # Set to evaluation mode
        
        # Pre-compute token IDs for true and false
        self.true_token_id = self.monot5_tokenizer.encode(" true")[0]
        self.false_token_id = self.monot5_tokenizer.encode(" false")[0]
        
        logger.info("MonoT5 model loaded successfully")
            
    def _load_test_data(self, test_data_path: str):
        """Load the test dataset."""
        logger.info(f"Loading test data from {test_data_path}")
        with open(test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        logger.info(f"Loaded {len(self.test_data)} test examples")
            
    def _load_faiss_index(self, index_path: str, mapping_path: str):
        """Load the FAISS index and mapping file."""
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
    
    def retrieve_initial_qa_pairs(self, questions: List[str]) -> List[List[Dict]]:
        """
        Retrieve initial set of QA pairs using FAISS (before reranking).
        
        Args:
            questions: List of question strings
            
        Returns:
            List of lists of retrieved QA pairs
        """
        # Generate embeddings for all questions
        query_embeddings = self.embed_questions(questions)
        
        # Search the FAISS index
        D, I = self.index.search(query_embeddings, self.initial_k + 5)  # Add buffer for filtering
        
        # Extract the QA pairs for each question
        all_retrieved_qa = []
        for i, indices in enumerate(I):
            retrieved_qa = []
            question = questions[i]
            
            for idx in indices:
                idx_key = str(idx)
                
                # Skip if index not found in mapping
                if idx_key not in self.index_to_qa:
                    continue
                    
                qa_pair = self.index_to_qa[idx_key]
                
                # Skip if the retrieved question is identical to the query
                if qa_pair['question'].strip() == question.strip():
                    continue
                    
                retrieved_qa.append(qa_pair)
                
                # Only keep initial_k pairs
                if len(retrieved_qa) >= self.initial_k:
                    break
                    
            all_retrieved_qa.append(retrieved_qa)
            
        return all_retrieved_qa

    def monot5_rerank(self, questions: List[str], retrieved_qa_list: List[List[Dict]]) -> List[List[Dict]]:
        """
        Rerank retrieved QA pairs using MonoT5 with improved relevance scoring.
        
        Args:
            questions: List of question strings
            retrieved_qa_list: List of lists of initially retrieved QA pairs
            
        Returns:
            List of lists of reranked QA pairs (top_k per question)
        """
        all_reranked_qa = []
        
        # Process each question separately
        for i, (question, retrieved_qa) in enumerate(zip(questions, retrieved_qa_list)):
            if not retrieved_qa:
                logger.warning(f"No QA pairs retrieved for question: {question}")
                all_reranked_qa.append([])
                continue
                
            # Prepare candidates for reranking with MonoT5
            scores = []
            
            # Process in smaller batches
            batch_size = min(self.batch_size, len(retrieved_qa))
            
            for j in range(0, len(retrieved_qa), batch_size):
                batch_qa = retrieved_qa[j:j+batch_size]
                
                # Create MonoT5 input format queries
                # MonoT5 standard format is "Query: [query] Document: [document] Relevant:"
                batch_inputs = []
                for qa in batch_qa:
                    # IMPORTANT: Use only the retrieved question as document
                    # This focuses the relevance comparison on question similarity
                    # which is more important for RAG retrieval quality
                    document = qa['question']
                    input_text = f"Query: {question} Document: {document} Relevant:"
                    batch_inputs.append(input_text)
                
                # Tokenize for MonoT5
                inputs = self.monot5_tokenizer(
                    batch_inputs, 
                    return_tensors="pt", 
                    max_length=self.max_input_length, 
                    truncation=True, 
                    padding=True
                ).to(self.device)
                
                # Get relevance scores from MonoT5
                with torch.no_grad():
                    # Generate the tokens and get logits for the first generated token only
                    outputs = self.monot5_model.generate(
                        **inputs,
                        max_length=3,  # Generate just enough for "true" or "false"
                        return_dict_in_generate=True,
                        output_scores=True,
                        num_beams=1,  # Use greedy decoding for consistency
                    )
                    
                    # Extract first-token logits for batch
                    first_token_logits = outputs.scores[0]  # Get logits of first token
                    
                    # Calculate "true" vs "false" probability for each item in batch
                    batch_scores = []
                    for k in range(len(batch_qa)):
                        logits = first_token_logits[k]
                        # Calculate score as difference between true and false probability
                        probs = torch.softmax(logits, dim=0)
                        
                        # Higher score for "true" prediction
                        true_prob = probs[self.true_token_id].item()
                        false_prob = probs[self.false_token_id].item()
                        relevance_score = true_prob - false_prob
                        batch_scores.append(relevance_score)
                    
                    scores.extend(batch_scores)
            
            # Combine scores with QA pairs and add original indices for stable sorting
            scored_qa = [(score, qa, idx) for idx, (score, qa) in enumerate(zip(scores, retrieved_qa))]
            
            # Sort by score in descending order, with original index as tiebreaker
            scored_qa.sort(key=lambda x: (x[0], -x[2]), reverse=True)
            
            # Take top_k after reranking
            reranked_qa = [qa for _, qa, _ in scored_qa[:self.top_k]]
            all_reranked_qa.append(reranked_qa)
            
            # Log scoring information for the first few examples
            if i == 0:  # Debug logging for first question only
                logger.info(f"\nMonoT5 Reranking Scores for: {question}")
                for score_idx, (score, qa, _) in enumerate(scored_qa[:min(5, len(scored_qa))]):
                    logger.info(f"Item {score_idx+1} - Score: {score:.4f}")
                    logger.info(f"  Q: {qa['question']}")
            
        return all_reranked_qa
    
    def retrieve_and_rerank_qa_pairs(self, questions: List[str]) -> List[List[Dict]]:
        """
        End-to-end retrieval and reranking of QA pairs.
        
        Args:
            questions: List of question strings
            
        Returns:
            List of lists of retrieved and reranked QA pairs
        """
        # Step 1: Initial retrieval with FAISS
        retrieved_qa_list = self.retrieve_initial_qa_pairs(questions)
        
        # Log some examples of initial retrieval
        if len(questions) > 0:
            logger.info("\n" + "="*50)
            logger.info(f"INITIAL FAISS RETRIEVAL EXAMPLE:")
            logger.info(f"Query: {questions[0]}")
            for i, qa in enumerate(retrieved_qa_list[0][:3]):  # Show top 3
                logger.info(f"Top {i+1}. Q: {qa['question']}")
                logger.info(f"       A: {qa['answer']}")
        
        # Step 2: Rerank with MonoT5
        reranked_qa_list = self.monot5_rerank(questions, retrieved_qa_list)
        
        # Log some examples after reranking
        if len(questions) > 0 and len(reranked_qa_list[0]) > 0:
            logger.info("\n" + "="*50)
            logger.info(f"AFTER MONOT5 RERANKING:")
            logger.info(f"Query: {questions[0]}")
            for i, qa in enumerate(reranked_qa_list[0]):
                logger.info(f"Top {i+1}. Q: {qa['question']}")
                logger.info(f"       A: {qa['answer']}")
            logger.info("="*50 + "\n")
        
        return reranked_qa_list
    
    def format_inputs(self, questions: List[str], all_retrieved_qa: List[List[Dict]]) -> List[str]:
        """
        Format inputs for the T5 model with improved context organization.
        
        Args:
            questions: List of question strings
            all_retrieved_qa: List of lists of retrieved QA pairs
            
        Returns:
            List of formatted input strings
        """
        formatted_inputs = []
        
        for question, retrieved_qa in zip(questions, all_retrieved_qa):
            # Create context string with better organization
            context_parts = []
            
            # Number each example and organize by relevance
            for i, qa in enumerate(retrieved_qa):
                # Add a reference number to each example
                context_parts.append(f"Example {i+1}:")
                context_parts.append(f"Patient Question: {qa['question']}")
                context_parts.append(f"Medical Answer: {qa['answer']}")
            
            # Join with newlines for better structure in the prompt
            context = " ".join(context_parts)
            
            # More structured prompt with explicit instructions
            input_text = (
                f"You are a Clinical Medical Expert. Use the context information to provide a detailed, "
                f"evidence-based answer to the medical question. Format your answer for a patient, using "
                f"clear language. Focus on accuracy and thoroughness. Your answer should be 3-4 sentences.\n\n"
                f"Context Information:\n{context}\n\n"
                f"Patient Question: {question}\n\n"
                f"Expert Answer:"
            )
            
            formatted_inputs.append(input_text)
            
        return formatted_inputs
    
    def generate_answers(self, formatted_inputs: List[str]) -> List[str]:
        """
        Generate answers using the T5 model with enhanced generation parameters.
        
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
            
            # Generate outputs with enhanced parameters for quality
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    min_length=50,            # Encourage longer answers
                    num_beams=5,              # Increase beam search for better quality
                    length_penalty=1.5,       # Encourage longer generations
                    no_repeat_ngram_size=3,   # Prevent repetition
                    early_stopping=False,     # Don't stop early
                    do_sample=True,           # Use sampling for more diverse answers
                    top_p=0.92,               # Nucleus sampling - slightly higher to allow more diversity
                    temperature=0.7           # Control randomness - lower for more focus
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
        
    def calculate_rouge(self, generated_answers: list, reference_answers: list) -> dict:
        """
        Calculate ROUGE scores for generated answers.
        
        Args:
            generated_answers: List of generated answer strings
            reference_answers: List of reference answer strings
            
        Returns:
            Dictionary with ROUGE scores
        """
        # Initialize the ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            # Calculate ROUGE scores
            scores = scorer.score(ref, gen)
            
            # Extract F1 scores
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # Calculate statistics
        results = {
            'rouge1_scores': rouge1_scores,
            'rouge2_scores': rouge2_scores, 
            'rougeL_scores': rougeL_scores,
            'avg_rouge1': float(np.mean(rouge1_scores)),
            'avg_rouge2': float(np.mean(rouge2_scores)),
            'avg_rougeL': float(np.mean(rougeL_scores))
        }
        
        return results
    
    def calculate_bert_score(self, generated_answers: list, reference_answers: list) -> dict:
        """
        Calculate BERTScore for generated answers.
        
        Args:
            generated_answers: List of generated answer strings
            reference_answers: List of reference answer strings
            
        Returns:
            Dictionary with BERTScore metrics
        """
        # Calculate BERTScore
        P, R, F1 = bert_score(generated_answers, reference_answers, lang="en", rescale_with_baseline=True)
        
        # Convert to Python lists
        precision_scores = P.tolist()
        recall_scores = R.tolist()
        f1_scores = F1.tolist()
        
        # Calculate statistics
        results = {
            'bertscore_precision_scores': precision_scores,
            'bertscore_recall_scores': recall_scores,
            'bertscore_f1_scores': f1_scores,
            'avg_bertscore_precision': float(np.mean(precision_scores)),
            'avg_bertscore_recall': float(np.mean(recall_scores)),
            'avg_bertscore_f1': float(np.mean(f1_scores))
        }
        
        return results
    
    def calculate_meteor(self, generated_answers: list, reference_answers: list) -> dict:
        """
        Calculate METEOR score for generated answers.
        
        Args:
            generated_answers: List of generated answer strings
            reference_answers: List of reference answer strings
            
        Returns:
            Dictionary with METEOR scores
        """
        # Load METEOR metric
        meteor = evaluate.load('meteor')
        
        # Calculate METEOR scores
        meteor_scores = []
        for gen, ref in zip(generated_answers, reference_answers):
            score = meteor.compute(predictions=[gen], references=[ref])['meteor']
            meteor_scores.append(score)
        
        # Calculate statistics
        results = {
            'meteor_scores': meteor_scores,
            'avg_meteor': float(np.mean(meteor_scores))
        }
        
        return results
    
    def evaluate(self, output_file: str = 'evaluation_results.json', sample_limit: int = 5000):
        """
        Run the full evaluation pipeline with improved MonoT5 reranking.
        
        Args:
            output_file: Path to save the evaluation results
            sample_limit: Maximum number of test samples to evaluate
        """
        logger.info("Starting evaluation with improved MonoT5 reranking")
        
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
            
            # Retrieve similar QA pairs and rerank them
            batch_retrieved_qa = self.retrieve_and_rerank_qa_pairs(batch_questions)
            
            # Format inputs
            batch_formatted_inputs = self.format_inputs(batch_questions, batch_retrieved_qa)
            
            # Generate answers
            batch_answers = self.generate_answers(batch_formatted_inputs)
            generated_answers.extend(batch_answers)
            
            # Print comprehensive information for each question in this batch
            for j in range(len(batch_questions)):
                logger.info(f"\n{'='*80}")
                logger.info(f"ORIGINAL QUESTION: {batch_questions[j]}")
                logger.info(f"{'='*80}")
                
                # Print retrieved QA pairs
                logger.info("RETRIEVED QA PAIRS (after MonoT5 reranking):")
                for idx, qa_pair in enumerate(batch_retrieved_qa[j]):
                    logger.info(f"\n--- Pair {idx+1} ---")
                    logger.info(f"Question: {qa_pair['question']}")
                    logger.info(f"Answer: {qa_pair['answer']}")
                
                # Print original and generated answers
                logger.info(f"\n{'='*80}")
                logger.info(f"ORIGINAL ANSWER: {batch_ref_answers[j]}")
                logger.info(f"{'='*80}")
                logger.info(f"GENERATED ANSWER: {batch_answers[j]}")
                logger.info(f"{'='*80}\n")
        
        # Calculate evaluation metrics
        logger.info("Calculating evaluation metrics...")
        
        # Initialize results dictionaries
        bleu_results = {}
        rouge_results = {}
        bertscore_results = {}
        meteor_results = {}
        
        # Calculate metrics based on what's available
        logger.info("Calculating BLEU scores")
        bleu_results = self.calculate_bleu(generated_answers, reference_answers)
        
        if self.use_rouge:
            logger.info("Calculating ROUGE scores")
            rouge_results = self.calculate_rouge(generated_answers, reference_answers)
        else:
            rouge_results = {
                'rouge1_scores': [0] * len(generated_answers),
                'rouge2_scores': [0] * len(generated_answers),
                'rougeL_scores': [0] * len(generated_answers),
                'avg_rouge1': 0.0,
                'avg_rouge2': 0.0,
                'avg_rougeL': 0.0
            }
            
        if self.use_bertscore:
            logger.info("Calculating BERTScore")
            bertscore_results = self.calculate_bert_score(generated_answers, reference_answers)
        else:
            bertscore_results = {
                'bertscore_precision_scores': [0] * len(generated_answers),
                'bertscore_recall_scores': [0] * len(generated_answers),
                'bertscore_f1_scores': [0] * len(generated_answers),
                'avg_bertscore_precision': 0.0,
                'avg_bertscore_recall': 0.0,
                'avg_bertscore_f1': 0.0
            }
            
        if self.use_meteor:
            logger.info("Calculating METEOR score")
            meteor_results = self.calculate_meteor(generated_answers, reference_answers)
        else:
            meteor_results = {
                'meteor_scores': [0] * len(generated_answers),
                'avg_meteor': 0.0
            }

        # Prepare full results
        for idx, (gen, ref) in enumerate(zip(generated_answers, reference_answers)):
            results['question'].append(questions[idx])
            results['generated_answer'].append(gen)
            results['reference_answer'].append(ref)
            results['bleu_score'].append(bleu_results['bleu_scores'][idx])
            
            # Add additional metrics
            if idx < len(rouge_results['rouge1_scores']):
                results['rouge1_score'].append(rouge_results['rouge1_scores'][idx])
                results['rouge2_score'].append(rouge_results['rouge2_scores'][idx])
                results['rougeL_score'].append(rouge_results['rougeL_scores'][idx])
            
            if idx < len(bertscore_results['bertscore_f1_scores']):
                results['bertscore_f1'].append(bertscore_results['bertscore_f1_scores'][idx])
            
            if idx < len(meteor_results['meteor_scores']):
                results['meteor_score'].append(meteor_results['meteor_scores'][idx])
        
        # Add summary statistics
        results['summary'] = {
            # BLEU scores
            'avg_bleu': bleu_results['avg_bleu'],
            'median_bleu': bleu_results['median_bleu'],
            'std_bleu': bleu_results['std_bleu'],
            'bleu_1': bleu_results['bleu_1'],
            'bleu_2': bleu_results['bleu_2'],
            'bleu_3': bleu_results['bleu_3'],
            'bleu_4': bleu_results['bleu_4'],
            
            # ROUGE scores
            'avg_rouge1': rouge_results['avg_rouge1'],
            'avg_rouge2': rouge_results['avg_rouge2'],
            'avg_rougeL': rouge_results['avg_rougeL'],
            
            # BERTScore
            'avg_bertscore_precision': bertscore_results['avg_bertscore_precision'],
            'avg_bertscore_recall': bertscore_results['avg_bertscore_recall'],
            'avg_bertscore_f1': bertscore_results['avg_bertscore_f1'],
            
            # METEOR
            'avg_meteor': meteor_results['avg_meteor']
        }
        
        # Log summary results
        logger.info(f"Evaluation Summary:")
        logger.info(f"BLEU Scores:")
        logger.info(f"  Average BLEU: {bleu_results['avg_bleu']:.4f}")
        logger.info(f"  BLEU-1: {bleu_results['bleu_1']:.4f}")
        logger.info(f"  BLEU-2: {bleu_results['bleu_2']:.4f}")
        logger.info(f"  BLEU-3: {bleu_results['bleu_3']:.4f}")
        logger.info(f"  BLEU-4: {bleu_results['bleu_4']:.4f}")
        
        logger.info(f"ROUGE Scores:")
        logger.info(f"  ROUGE-1: {rouge_results['avg_rouge1']:.4f}")
        logger.info(f"  ROUGE-2: {rouge_results['avg_rouge2']:.4f}")
        logger.info(f"  ROUGE-L: {rouge_results['avg_rougeL']:.4f}")
        
        logger.info(f"BERTScore:")
        logger.info(f"  Precision: {bertscore_results['avg_bertscore_precision']:.4f}")
        logger.info(f"  Recall: {bertscore_results['avg_bertscore_recall']:.4f}")
        logger.info(f"  F1: {bertscore_results['avg_bertscore_f1']:.4f}")
        
        logger.info(f"METEOR Score: {meteor_results['avg_meteor']:.4f}")
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results

def main():
    """Main function to run the evaluation with improved MonoT5 reranking."""
    parser = argparse.ArgumentParser(description="Evaluate T5 model on medical QA dataset with MonoT5 reranking")
    parser.add_argument("--t5_model_name", type=str, default="google/flan-t5-base", 
                        help="HuggingFace T5 model name")
    parser.add_argument("--monot5_model_name", type=str, default="castorini/monot5-base-msmarco", 
                        help="HuggingFace model name for MonoT5 reranker")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Path to test data JSON file")
    parser.add_argument("--faiss_index", type=str, required=True, 
                        help="Path to FAISS index file")
    parser.add_argument("--mapping_file", type=str, required=True, 
                        help="Path to index mapping pickle file")
    parser.add_argument("--output_file", type=str, default="evaluation_results_monot5.json", 
                        help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing")
    parser.add_argument("--initial_k", type=int, default=16, 
                        help="Number of QA pairs to initially retrieve")
    parser.add_argument("--top_k", type=int, default=4, 
                        help="Number of QA pairs to keep after reranking")
    parser.add_argument("--sample_limit", type=int, default=5000,
                        help="Maximum number of test samples to evaluate")
    parser.add_argument("--use_auth_token", action="store_true",
                        help="Use HuggingFace auth token for private models")
    parser.add_argument("--experiment_name", type=str, default="monot5_reranking",
                        help="Name for the experiment, used for logging")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MedicalQAEvaluator(
        t5_model_name=args.t5_model_name,
        monot5_model_name=args.monot5_model_name,
        test_data_path=args.test_data,
        faiss_index_path=args.faiss_index,
        mapping_file_path=args.mapping_file,
        batch_size=args.batch_size,
        initial_k=args.initial_k,
        top_k=args.top_k,
        use_auth_token=args.use_auth_token
    )
    # Run evaluation with sample limit
    evaluator.evaluate(args.output_file, args.sample_limit)

if __name__ == "__main__":
    main()