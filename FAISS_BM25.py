import json
import os
import faiss
import pickle
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple, Optional
import argparse
import logging
from collections import defaultdict, Counter
import math
import re
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

class BM25:
    """BM25 implementation for document scoring."""
    
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25):
        """
        Initialize BM25 parameters.
        
        Args:
            k1: Term saturation parameter
            b: Length normalization parameter
            epsilon: Additive smoothing parameter
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus = []
        self.doc_freqs = Counter()
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.N = 0
        self.initialized = False
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """
        Preprocess text for BM25 indexing or search.
        
        Args:
            text: Text to preprocess
        
        Returns:
            List of tokens
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphanumeric tokens
        tokens = [token for token in tokens if token not in self.stop_words and re.match(r'^[a-zA-Z0-9]+$', token)]
        
        return tokens
    
    def fit(self, corpus):
        """
        Fit BM25 model on a corpus.
        
        Args:
            corpus: List of documents
        """
        # Preprocess corpus
        tokenized_corpus = [self.preprocess(doc) for doc in corpus]
        self.corpus = tokenized_corpus
        self.N = len(tokenized_corpus)
        
        # Calculate document frequencies
        for doc in tokenized_corpus:
            self.doc_len.append(len(doc))
            
            # Count unique terms in document
            for term in set(doc):
                self.doc_freqs[term] += 1
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / self.N
        
        # Calculate IDF for each term
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
        
        self.initialized = True
    
    def get_scores(self, query):
        """
        Score documents in corpus for a given query.
        
        Args:
            query: Query string
        
        Returns:
            List of scores for each document
        """
        if not self.initialized:
            raise ValueError("BM25 model has not been fit on a corpus.")
        
        # Preprocess query
        query_tokens = self.preprocess(query)
        
        # Calculate scores
        scores = np.zeros(self.N)
        
        for term in query_tokens:
            if term not in self.idf:
                continue
                
            # Calculate query term weight (IDF)
            q_weight = self.idf[term]
            
            # Calculate score for each document
            for i, doc in enumerate(self.corpus):
                if term not in doc:
                    continue
                
                # Calculate term frequency in document
                doc_term_count = doc.count(term)
                
                # Calculate BM25 score for this term
                doc_len_norm = (1 - self.b) + self.b * (self.doc_len[i] / self.avgdl)
                term_score = q_weight * ((doc_term_count * (self.k1 + 1)) / 
                                        (doc_term_count + self.k1 * doc_len_norm))
                
                scores[i] += term_score
        
        return scores

class MedicalQAEvaluator:
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-base",
        test_data_path: str = None,
        faiss_index_path: str = None,
        mapping_file_path: str = None,
        batch_size: int = 8,
        initial_k: int = 16,  # Retrieve more candidates for reranking
        top_k: int = 4,       # Final number of QA pairs after reranking
        max_input_length: int = 512,
        max_output_length: int = 256,  # Increased from 128 to allow for longer answers
        use_auth_token: bool = False,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """
        Initialize the Medical QA Evaluator with BM25 reranking.
        
        Args:
            t5_model_name: HuggingFace model name for T5
            test_data_path: Path to test data JSON file
            faiss_index_path: Path to FAISS index file
            mapping_file_path: Path to index mapping pickle file
            batch_size: Batch size for processing
            initial_k: Number of QA pairs to initially retrieve (before reranking)
            top_k: Number of QA pairs to keep after reranking
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            use_auth_token: Whether to use HuggingFace auth token for private models
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.batch_size = batch_size
        self.initial_k = initial_k
        self.top_k = top_k
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_auth_token = use_auth_token
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models and data
        self._load_t5_model(t5_model_name)
        self._load_biobert_model()
        self._init_bm25()
        
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
    
    def _init_bm25(self):
        """Initialize the BM25 model for reranking."""
        try:
            logger.info("Initializing BM25 reranker")
            self.bm25 = BM25(k1=self.bm25_k1, b=self.bm25_b)
            logger.info("BM25 reranker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BM25 reranker: {e}")
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
        D, I = self.index.search(query_embeddings, self.initial_k + 1)  # +1 to handle self-matches
        
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
                
                # Only keep initial_k pairs
                if len(retrieved_qa) >= self.initial_k:
                    break
                    
            all_retrieved_qa.append(retrieved_qa)
            
        return all_retrieved_qa
    
    def bm25_rerank(self, questions: List[str], retrieved_qa_list: List[List[Dict]]) -> List[List[Dict]]:
        """
        Rerank retrieved QA pairs using BM25.
        
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
                
            # Prepare corpus for BM25
            corpus = []
            for qa in retrieved_qa:
                # Combine question and answer for reranking
                document = qa['question'] + " " + qa['answer']
                corpus.append(document)
            
            # Fit BM25 on this corpus
            self.bm25.fit(corpus)
            
            # Get scores for the query
            scores = self.bm25.get_scores(question)
            
            # Combine scores with QA pairs and add original indices for stable sorting
            scored_qa = [(score, qa, idx) for idx, (score, qa) in enumerate(zip(scores, retrieved_qa))]
            
            # Sort by score in descending order, with original index as tiebreaker
            scored_qa.sort(key=lambda x: (x[0], -x[2]), reverse=True)
            
            # Take top_k after reranking
            reranked_qa = [qa for _, qa, _ in scored_qa[:self.top_k]]
            all_reranked_qa.append(reranked_qa)
            
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
        
        # Step 2: Rerank with BM25
        reranked_qa_list = self.bm25_rerank(questions, retrieved_qa_list)
        
        # Log some examples after reranking
        if len(questions) > 0 and len(reranked_qa_list[0]) > 0:
            logger.info("\n" + "="*50)
            logger.info(f"AFTER BM25 RERANKING:")
            logger.info(f"Query: {questions[0]}")
            for i, qa in enumerate(reranked_qa_list[0]):
                logger.info(f"Top {i+1}. Q: {qa['question']}")
                logger.info(f"       A: {qa['answer']}")
            logger.info("="*50 + "\n")
        
        return reranked_qa_list
    
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
            try:
                # Calculate ROUGE scores
                scores = scorer.score(ref, gen)
                
                # Extract F1 scores
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception as e:
                logger.warning(f"Error calculating ROUGE score: {e}")
                rouge1_scores.append(0)
                rouge2_scores.append(0)
                rougeL_scores.append(0)
        
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
        try:
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
            
        except Exception as e:
            logger.warning(f"Error calculating BERTScore: {e}")
            # Return empty results on error
            results = {
                'bertscore_precision_scores': [],
                'bertscore_recall_scores': [],
                'bertscore_f1_scores': [],
                'avg_bertscore_precision': 0.0,
                'avg_bertscore_recall': 0.0,
                'avg_bertscore_f1': 0.0
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
        try:
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
            
        except Exception as e:
            logger.warning(f"Error calculating METEOR score: {e}")
            results = {
                'meteor_scores': [],
                'avg_meteor': 0.0
            }
            
        return results
    
    def evaluate(self, output_file: str = 'evaluation_results.json', sample_limit: int = 5000):
        """
        Run the full evaluation pipeline with BM25 reranking.
        
        Args:
            output_file: Path to save the evaluation results
            sample_limit: Maximum number of test samples to evaluate
        """
        logger.info("Starting evaluation with BM25 reranking")
        
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
                logger.info("RETRIEVED QA PAIRS (after BM25 reranking):")
                for idx, qa_pair in enumerate(batch_retrieved_qa[j]):
                    logger.info(f"\n--- Pair {idx+1} ---")
                    logger.info(f"Question: {qa_pair['question']}")
                
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
        
        # Calculate selected metrics
        logger.info("Calculating BLEU scores")
        bleu_results = self.calculate_bleu(generated_answers, reference_answers)
        logger.info("Calculating ROUGE scores")
        rouge_results = self.calculate_rouge(generated_answers, reference_answers)
        logger.info("Calculating BERTScore")
        bertscore_results = self.calculate_bert_score(generated_answers, reference_answers)
        logger.info("Calculating METEOR score")
        meteor_results = self.calculate_meteor(generated_answers, reference_answers)
        
        # Prepare full results
        for idx, (gen, ref) in enumerate(zip(generated_answers, reference_answers)):
            results['question'].append(questions[idx])
            results['generated_answer'].append(gen)
            results['reference_answer'].append(ref)
            results['bleu_score'].append(bleu_results['bleu_scores'][idx])
            
            # Add additional metrics when available
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
            # BM25 parameters
            'bm25_k1': self.bm25_k1,
            'bm25_b': self.bm25_b,
            
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
    parser = argparse.ArgumentParser(description="Evaluate T5 model on medical QA dataset with BM25 reranking")
    parser.add_argument("--t5_model_name", type=str, default="google/flan-t5-base", 
                        help="HuggingFace T5 model name")
    parser.add_argument("--test_data", type=str, required=True, 
                        help="Path to test data JSON file")
    parser.add_argument("--faiss_index", type=str, required=True, 
                        help="Path to FAISS index file")
    parser.add_argument("--mapping_file", type=str, required=True, 
                        help="Path to index mapping pickle file")
    parser.add_argument("--output_file", type=str, default="new_output/T5_FAISS_BM25_evaluation_results.json", 
                        help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for processing")
    parser.add_argument("--initial_k", type=int, default=16, 
                        help="Number of QA pairs to initially retrieve")
    parser.add_argument("--top_k", type=int, default=4, 
                        help="Number of QA pairs to keep after reranking")
    parser.add_argument("--bm25_k1", type=float, default=1.5, 
                        help="BM25 k1 parameter (term saturation)")
    parser.add_argument("--bm25_b", type=float, default=0.75, 
                        help="BM25 b parameter (length normalization)")
    parser.add_argument("--sample_limit", type=int, default=5000,
                        help="Maximum number of test samples to evaluate")
    parser.add_argument("--use_auth_token", action="store_true",
                        help="Use HuggingFace auth token for private models")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MedicalQAEvaluator(
        t5_model_name=args.t5_model_name,
        test_data_path=args.test_data,
        faiss_index_path=args.faiss_index,
        mapping_file_path=args.mapping_file,
        batch_size=args.batch_size,
        initial_k=args.initial_k,
        top_k=args.top_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        use_auth_token=args.use_auth_token
    )
    
    # Run evaluation with sample limit
    evaluator.evaluate(args.output_file, args.sample_limit)

if __name__ == "__main__":
    main()