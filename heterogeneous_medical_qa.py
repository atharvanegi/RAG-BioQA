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
from typing import List, Dict, Any, Tuple, Optional, Union
import argparse
import logging
from collections import defaultdict
import datetime
import requests
from dateutil.parser import parse as date_parse
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("heterogeneous_qa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KnowledgeSource:
    """Base class for all knowledge sources"""
    
    def __init__(self, name: str, priority: float = 1.0, recency_weight: float = 0.5):
        """
        Initialize a knowledge source
        
        Args:
            name: Name of the knowledge source
            priority: Base priority/importance of this source (1.0 is neutral)
            recency_weight: How much to weight recency for this source (0.0-1.0)
        """
        self.name = name
        self.priority = priority
        self.recency_weight = recency_weight
        
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search this knowledge source for relevant documents
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of documents with metadata
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    def get_document_date(self, doc: Dict) -> datetime.datetime:
        """
        Extract the publication date from a document
        
        Args:
            doc: Document dictionary
            
        Returns:
            Publication date as datetime object
        """
        raise NotImplementedError("Subclasses must implement get_document_date()")

class FAISSKnowledgeSource(KnowledgeSource):
    """Knowledge source using FAISS index"""
    
    def __init__(
        self, 
        name: str,
        index_path: str,
        mapping_file_path: str,
        biobert_model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        priority: float = 1.0,
        recency_weight: float = 0.5,
        device: str = None
    ):
        """
        Initialize FAISS knowledge source
        
        Args:
            name: Name of the knowledge source
            index_path: Path to FAISS index file
            mapping_file_path: Path to index mapping pickle file
            biobert_model_name: Name of BioBERT model to use
            priority: Base priority of this source
            recency_weight: Weight of recency for this source
            device: Device to use for BioBERT
        """
        super().__init__(name, priority, recency_weight)
        
        # Set device
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BioBERT model
        logger.info(f"Loading BioBERT model {biobert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
        self.model = AutoModel.from_pretrained(biobert_model_name).to(self.device)
        self.model.eval()
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        # Load mapping file
        logger.info(f"Loading index mapping from {mapping_file_path}")
        with open(mapping_file_path, 'rb') as f:
            mapping = pickle.load(f)
        
        # Ensure all keys are strings for consistent lookup
        if isinstance(next(iter(mapping['index_to_qa'].keys()), 0), int):
            self.index_to_qa = {str(k): v for k, v in mapping['index_to_qa'].items()}
        else:
            self.index_to_qa = mapping['index_to_qa']
            
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.index_to_qa)} mappings")
    
    def embed_question(self, question: str) -> np.ndarray:
        """Generate BioBERT embedding for a question"""
        inputs = self.tokenizer(
            question, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.sum(attention_mask, 1)
            embedding = sum_embeddings / sum_mask
            
        # Convert to numpy and normalize
        embedding = embedding.cpu().numpy()
        faiss.normalize_L2(embedding)
        return embedding
        
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search FAISS index for similar questions"""
        # Generate query embedding
        query_embedding = self.embed_question(query)
        
        # Search index
        D, I = self.index.search(query_embedding, top_k + 5)  # Get extra results to handle missing mappings
        
        # Extract documents
        results = []
        for i, idx in enumerate(I[0]):
            idx_key = str(idx)
            
            # Skip if not in mapping
            if idx_key not in self.index_to_qa:
                continue
                
            qa_pair = self.index_to_qa[idx_key]
            
            # Skip exact matches to query
            if qa_pair['question'].strip() == query.strip():
                continue
                
            # Add metadata
            doc = qa_pair.copy()
            doc['score'] = float(D[0][i])
            doc['source'] = self.name
            doc['source_id'] = idx_key
            
            results.append(doc)
            
            # Stop after getting top_k results
            if len(results) >= top_k:
                break
                
        return results
    
    def get_document_date(self, doc: Dict) -> datetime.datetime:
        """Extract date from document"""
        # Try to get date from metadata or use default
        if 'date' in doc:
            try:
                return date_parse(doc['date'])
            except (ValueError, TypeError):
                pass
                
        # If publication date not available, use a default
        return datetime.datetime(2020, 1, 1)  # Default date if not provided

class PubMedKnowledgeSource(KnowledgeSource):
    """Knowledge source using PubMed API"""
    
    def __init__(
        self, 
        name: str = "PubMed",
        priority: float = 1.2,  # Higher priority for peer-reviewed content
        recency_weight: float = 0.8,  # Strong recency weight for research
        api_key: str = None,
        max_age_years: int = 5  # Only consider papers from last 5 years by default
    ):
        """
        Initialize PubMed knowledge source
        
        Args:
            name: Name of the knowledge source
            priority: Base priority of this source
            recency_weight: Weight of recency for this source
            api_key: NCBI API key (optional)
            max_age_years: Maximum age of papers to consider
        """
        super().__init__(name, priority, recency_weight)
        self.api_key = api_key
        self.max_age_years = max_age_years
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search PubMed for relevant articles"""
        # Construct search query with date restriction
        current_year = datetime.datetime.now().year
        min_year = current_year - self.max_age_years
        date_query = f"{min_year}:{current_year}[pdat]"
        
        # Prepare API parameters
        params = {
            "db": "pubmed",
            "term": f"{query} AND {date_query}",
            "retmode": "json",
            "retmax": top_k,
            "sort": "relevance"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        # First get IDs of matching articles
        search_url = f"{self.base_url}esearch.fcgi"
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            search_result = response.json()
            
            if "esearchresult" not in search_result or "idlist" not in search_result["esearchresult"]:
                logger.warning(f"No results found in PubMed for query: {query}")
                return []
                
            pmids = search_result["esearchresult"]["idlist"]
            
            if not pmids:
                logger.info(f"No PubMed articles found for query: {query}")
                return []
                
            # Fetch details for these IDs
            fetch_url = f"{self.base_url}esummary.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json"
            }
            
            if self.api_key:
                fetch_params["api_key"] = self.api_key
                
            fetch_response = requests.get(fetch_url, fetch_params)
            fetch_response.raise_for_status()
            
            details = fetch_response.json()
            
            # Process results
            results = []
            if "result" not in details:
                logger.warning("No details found in PubMed API response")
                return []
                
            for pmid in pmids:
                if pmid not in details["result"]:
                    continue
                    
                article = details["result"][pmid]
                
                # Extract article info
                title = article.get("title", "")
                authors = ", ".join([author.get("name", "") for author in article.get("authors", [])])
                
                # Get publication date
                pub_date = None
                if "pubdate" in article:
                    try:
                        pub_date = date_parse(article["pubdate"])
                    except (ValueError, TypeError):
                        pub_date = None
                        
                journal = article.get("fulljournalname", "")
                
                # Create document
                doc = {
                    "question": title,
                    "answer": f"According to {journal}, {title} by {authors}.",
                    "source": self.name,
                    "source_id": pmid,
                    "date": article.get("pubdate", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "journal": journal,
                    "authors": authors,
                    "score": 1.0  # Base score, will be adjusted by time-aware ranking
                }
                
                results.append(doc)
                
            return results
                
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def get_document_date(self, doc: Dict) -> datetime.datetime:
        """Extract date from PubMed document"""
        if "date" in doc and doc["date"]:
            try:
                return date_parse(doc["date"])
            except (ValueError, TypeError):
                pass
                
        # Default date if parsing fails
        return datetime.datetime(2020, 1, 1)

class ClinicalGuidelinesKnowledgeSource(KnowledgeSource):
    """Simulated knowledge source for clinical guidelines"""
    
    def __init__(
        self, 
        name: str = "ClinicalGuidelines",
        guidelines_file: str = "clinical_guidelines.json",
        priority: float = 1.5,  # Highest priority for official guidelines
        recency_weight: float = 0.6  # Moderate recency weight
    ):
        """
        Initialize clinical guidelines knowledge source
        
        Args:
            name: Name of the knowledge source
            guidelines_file: Path to JSON file containing guidelines
            priority: Base priority of this source
            recency_weight: Weight of recency for this source
        """
        super().__init__(name, priority, recency_weight)
        self.guidelines_file = guidelines_file
        self._load_guidelines()
        
    def _load_guidelines(self):
        """Load guidelines from file or initialize empty if file doesn't exist"""
        try:
            if os.path.exists(self.guidelines_file):
                with open(self.guidelines_file, 'r', encoding='utf-8') as f:
                    self.guidelines = json.load(f)
                logger.info(f"Loaded {len(self.guidelines)} guidelines from {self.guidelines_file}")
            else:
                # Initialize with sample guidelines if file doesn't exist
                self.guidelines = self._create_sample_guidelines()
                with open(self.guidelines_file, 'w', encoding='utf-8') as f:
                    json.dump(self.guidelines, f, indent=2)
                logger.info(f"Created sample guidelines file at {self.guidelines_file}")
        except Exception as e:
            logger.error(f"Error loading guidelines: {e}")
            self.guidelines = self._create_sample_guidelines()
            
    def _create_sample_guidelines(self) -> List[Dict]:
        """Create sample guidelines for demonstration purposes"""
        return [
            {
                "title": "Hypertension Management Guidelines",
                "organization": "American Heart Association",
                "content": "Adults with confirmed hypertension should be treated with anti-hypertensive medication based on their cardiovascular risk profile.",
                "date": "2023-01-15",
                "url": "https://www.heart.org/guidelines/hypertension",
                "keywords": ["hypertension", "blood pressure", "antihypertensive", "cardiovascular risk"]
            },
            {
                "title": "Type 2 Diabetes Management",
                "organization": "American Diabetes Association",
                "content": "Metformin remains the preferred initial pharmacologic agent for the treatment of type 2 diabetes.",
                "date": "2022-12-10",
                "url": "https://diabetes.org/guidelines",
                "keywords": ["diabetes", "type 2 diabetes", "metformin", "blood glucose"]
            },
            {
                "title": "Asthma Treatment Guidelines",
                "organization": "Global Initiative for Asthma",
                "content": "Inhaled corticosteroids are the most effective medications for long-term asthma management.",
                "date": "2023-03-22",
                "url": "https://ginasthma.org/guidelines",
                "keywords": ["asthma", "inhaled corticosteroids", "asthma management", "bronchodilators"]
            }
        ]
        
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search guidelines for relevant content"""
        # Simple keyword matching for demonstration
        query_terms = set(query.lower().split())
        
        scored_guidelines = []
        for guideline in self.guidelines:
            # Calculate simple keyword overlap score
            keyword_matches = sum(1 for keyword in guideline["keywords"] if keyword in query_terms)
            
            # Calculate BM25-inspired score based on title and content
            title_terms = set(guideline["title"].lower().split())
            content_terms = set(guideline["content"].lower().split())
            
            title_overlap = len(query_terms.intersection(title_terms)) / (len(query_terms) + 0.5)
            content_overlap = len(query_terms.intersection(content_terms)) / (len(query_terms) + 0.5)
            
            # Combined score
            score = keyword_matches * 0.4 + title_overlap * 0.4 + content_overlap * 0.2
            
            if score > 0:
                doc = {
                    "question": guideline["title"],
                    "answer": guideline["content"],
                    "source": self.name,
                    "source_id": str(hash(guideline["title"])),
                    "date": guideline["date"],
                    "organization": guideline["organization"],
                    "url": guideline["url"],
                    "score": score
                }
                scored_guidelines.append(doc)
                
        # Sort by score and take top_k
        scored_guidelines.sort(key=lambda x: x["score"], reverse=True)
        return scored_guidelines[:top_k]
    
    def get_document_date(self, doc: Dict) -> datetime.datetime:
        """Extract date from guideline document"""
        if "date" in doc and doc["date"]:
            try:
                return date_parse(doc["date"])
            except (ValueError, TypeError):
                pass
                
        # Default date if parsing fails
        return datetime.datetime(2022, 1, 1)

class MedicalTextbookKnowledgeSource(KnowledgeSource):
    """Simulated knowledge source for medical textbooks"""
    
    def __init__(
        self, 
        name: str = "MedicalTextbook",
        textbook_file: str = "medical_textbook.json",
        priority: float = 0.8,  # Lower priority than guidelines but stable information
        recency_weight: float = 0.3  # Low recency weight for foundational knowledge
    ):
        """
        Initialize medical textbook knowledge source
        
        Args:
            name: Name of the knowledge source
            textbook_file: Path to JSON file containing textbook content
            priority: Base priority of this source
            recency_weight: Weight of recency for this source
        """
        super().__init__(name, priority, recency_weight)
        self.textbook_file = textbook_file
        self._load_textbook()
        
    def _load_textbook(self):
        """Load textbook content from file or initialize empty if file doesn't exist"""
        try:
            if os.path.exists(self.textbook_file):
                with open(self.textbook_file, 'r', encoding='utf-8') as f:
                    self.textbook = json.load(f)
                logger.info(f"Loaded {len(self.textbook)} textbook entries from {self.textbook_file}")
            else:
                # Initialize with sample textbook if file doesn't exist
                self.textbook = self._create_sample_textbook()
                with open(self.textbook_file, 'w', encoding='utf-8') as f:
                    json.dump(self.textbook, f, indent=2)
                logger.info(f"Created sample textbook file at {self.textbook_file}")
        except Exception as e:
            logger.error(f"Error loading textbook: {e}")
            self.textbook = self._create_sample_textbook()
            
    def _create_sample_textbook(self) -> List[Dict]:
        """Create sample textbook entries for demonstration purposes"""
        return [
            {
                "title": "Anatomy of the Heart",
                "textbook": "Gray's Anatomy",
                "content": "The heart is a muscular organ located in the chest cavity between the lungs. It consists of four chambers: right atrium, right ventricle, left atrium, and left ventricle.",
                "edition": "42nd Edition",
                "date": "2020-05-15",
                "keywords": ["heart", "anatomy", "cardiac", "chambers", "atrium", "ventricle"]
            },
            {
                "title": "Pathophysiology of Diabetes Mellitus",
                "textbook": "Harrison's Principles of Internal Medicine",
                "content": "Diabetes mellitus is characterized by hyperglycemia resulting from defects in insulin secretion, insulin action, or both. Type 1 diabetes is caused by pancreatic beta cell destruction, while Type 2 diabetes involves insulin resistance.",
                "edition": "21st Edition",
                "date": "2022-01-10",
                "keywords": ["diabetes", "hyperglycemia", "insulin", "pathophysiology", "pancreas"]
            },
            {
                "title": "Classification of Antibiotics",
                "textbook": "Goodman & Gilman's The Pharmacological Basis of Therapeutics",
                "content": "Antibiotics are classified based on their mechanism of action: cell wall synthesis inhibitors (penicillins, cephalosporins), protein synthesis inhibitors (macrolides, tetracyclines), nucleic acid synthesis inhibitors (fluoroquinolones), and metabolic pathway inhibitors (sulfonamides).",
                "edition": "14th Edition",
                "date": "2021-08-22",
                "keywords": ["antibiotics", "antimicrobial", "penicillins", "cephalosporins", "macrolides"]
            }
        ]
        
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search textbook for relevant content"""
        # Simple keyword matching for demonstration
        query_terms = set(query.lower().split())
        
        scored_entries = []
        for entry in self.textbook:
            # Calculate simple score based on keyword overlap
            keywords = set(entry["keywords"])
            keyword_matches = sum(1 for term in query_terms if any(keyword.lower().find(term) >= 0 for keyword in keywords))
            
            # Calculate score based on title and content overlap
            title_terms = set(entry["title"].lower().split())
            content_words = entry["content"].lower()
            
            title_overlap = len(query_terms.intersection(title_terms)) / (len(query_terms) + 0.5)
            content_score = sum(1 for term in query_terms if term in content_words) / len(query_terms)
            
            # Combined score
            score = keyword_matches * 0.3 + title_overlap * 0.4 + content_score * 0.3
            
            if score > 0:
                doc = {
                    "question": entry["title"],
                    "answer": entry["content"],
                    "source": self.name,
                    "source_id": str(hash(entry["title"])),
                    "date": entry["date"],
                    "textbook": entry["textbook"],
                    "edition": entry["edition"],
                    "score": score
                }
                scored_entries.append(doc)
                
        # Sort by score and take top_k
        scored_entries.sort(key=lambda x: x["score"], reverse=True)
        return scored_entries[:top_k]
    
    def get_document_date(self, doc: Dict) -> datetime.datetime:
        """Extract date from textbook document"""
        if "date" in doc and doc["date"]:
            try:
                return date_parse(doc["date"])
            except (ValueError, TypeError):
                pass
                
        # Default date for textbooks
        return datetime.datetime(2020, 1, 1)

class HeterogeneousMedicalQA:
    """Medical QA system with heterogeneous knowledge integration and time-aware retrieval"""
    
    def __init__(
        self,
        knowledge_sources: List[KnowledgeSource],
        t5_model_name: str = "google/flan-t5-base",
        max_input_length: int = 1024,
        max_output_length: int = 256,
        top_k_per_source: int = 5,
        final_top_k: int = 8,
        recency_half_life_days: int = 365,  # Weight halves every year
        device: str = None,
        use_auth_token: bool = False
    ):
        """
        Initialize heterogeneous medical QA system
        
        Args:
            knowledge_sources: List of knowledge sources to use
            t5_model_name: Name of T5 model for answer generation
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            top_k_per_source: Number of documents to retrieve from each source
            final_top_k: Final number of documents to use after reranking
            recency_half_life_days: Number of days after which recency score halves
            device: Device to use for models
            use_auth_token: Whether to use HuggingFace auth token
        """
        self.knowledge_sources = knowledge_sources
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.top_k_per_source = top_k_per_source
        self.final_top_k = final_top_k
        self.recency_half_life_days = recency_half_life_days
        
        # Set device
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load T5 model for answer generation
        logger.info(f"Loading T5 model: {t5_model_name}")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            t5_model_name,
            use_auth_token=use_auth_token
        )
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model_name,
            use_auth_token=use_auth_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Initialize NLTK for BLEU scoring (if evaluating)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Initialize BLEU smoothing
        self.smoother = SmoothingFunction().method1
        
    def retrieve_from_sources(self, query: str) -> List[Dict]:
        """
        Retrieve documents from all knowledge sources
        
        Args:
            query: Query string
            
        Returns:
            List of documents from all sources
        """
        all_documents = []
        
        # Retrieve from each source
        for source in self.knowledge_sources:
            try:
                logger.info(f"Retrieving from {source.name}...")
                documents = source.search(query, self.top_k_per_source)
                logger.info(f"Retrieved {len(documents)} documents from {source.name}")
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error retrieving from {source.name}: {e}")
                
        return all_documents
    
    def apply_time_aware_ranking(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Apply time-aware ranking to documents
        
        Args:
            query: Original query string
            documents: List of documents from all sources
            
        Returns:
            Ranked list of documents
        """
        if not documents:
            return []
            
        # Current time for recency calculations
        now = datetime.datetime.now()
        
        # Calculate time score for each document
        for doc in documents:
            # Get source
            source_name = doc["source"]
            source = next((s for s in self.knowledge_sources if s.name == source_name), None)
            
            if not source:
                # Default values if source not found
                base_priority = 1.0
                recency_weight = 0.5
            else:
                base_priority = source.priority
                recency_weight = source.recency_weight
                
            # Get document date
            try:
                if source:
                    doc_date = source.get_document_date(doc)
                else:
                    # Default parsing if source not available
                    doc_date = date_parse(doc.get("date", "2020-01-01"))
            except (ValueError, TypeError):
                # Default date if parsing fails
                doc_date = datetime.datetime(2020, 1, 1)
                
            # Calculate days since publication
            days_old = max(0, (now - doc_date).days)
            
            # Calculate recency score using half-life formula
            # 1.0 for current documents, 0.5 after half_life days, etc.
            recency_score = pow(0.5, days_old / self.recency_half_life_days)
            
            # Calculate final score as weighted combination
            # Base score * source priority * recency factor
            base_score = doc.get("score", 1.0)
            
            # Recency component
            recency_component = recency_weight * recency_score
            
            # Source priority component
            priority_component = (1.0 - recency_weight) * base_priority
            
            # Final score combines base score with time and priority weights
            final_score = base_score * (recency_component + priority_component)
            
            # Update document with scoring details
            doc["final_score"] = final_score
            doc["recency_score"] = recency_score
            doc["days_old"] = days_old
            doc["date_parsed"] = doc_date.isoformat()
            
        # Rank documents by final score
        ranked_documents = sorted(documents, key=lambda x: x["final_score"], reverse=True)
        
        # Take top_k documents for final result
        return ranked_documents[:self.final_top_k]
    
    def format_context(self, documents: List[Dict]) -> str:
        """
        Format documents into context string for T5
        
        Args:
            documents: List of ranked documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Extract basic information
            question = doc.get("question", "")
            answer = doc.get("answer", "")
            source = doc.get("source", "Unknown")
            
            # Format date
            try:
                if "date_parsed" in doc:
                    date_obj = date_parse(doc["date_parsed"])
                    date_str = date_obj.strftime("%Y-%m-%d")
                elif "date" in doc:
                    date_obj = date_parse(doc["date"])
                    date_str = date_obj.strftime("%Y-%m-%d")
                else:
                    date_str = "Unknown date"
            except (ValueError, TypeError):
                date_str = "Unknown date"
                
            # Format additional info
            additional_info = []
            if "organization" in doc:
                additional_info.append(f"Organization: {doc['organization']}")
            if "journal" in doc:
                additional_info.append(f"Journal: {doc['journal']}")
            if "textbook" in doc:
                additional_info.append(f"Textbook: {doc['textbook']}")
                if "edition" in doc:
                    additional_info.append(f"Edition: {doc['edition']}")
                    
            additional_str = ", ".join(additional_info)
            
            # Format document
            doc_str = f"[Document {i+1}] Source: {source} ({date_str})"
            if additional_str:
                doc_str += f", {additional_str}"
            doc_str += f"\nQuestion: {question}\nAnswer: {answer}\n"
            
            context_parts.append(doc_str)
            
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using T5 model
        
        Args:
            query: Original query string
            context: Formatted context string
            
        Returns:
            Generated answer
        """
        # Format input for T5
        input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize input
        inputs = self.t5_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
        # Decode output
        answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    
    def answer_question(self, query: str) -> Dict:
        """
        End-to-end pipeline to answer a medical question
        
        Args:
            query: Question string
            
        Returns:
            Dictionary with answer and supporting information
        """
        # Step 1: Retrieve from heterogeneous sources
        logger.info(f"Processing query: {query}")
        documents = self.retrieve_from_sources(query)
        
        if not documents:
            logger.warning("No documents retrieved from any source")
            return {
                "query": query,
                "answer": "I don't have enough information to answer this question.",
                "documents": [],
                "sources_used": []
            }
            
        # Step 2: Apply time-aware ranking
        logger.info("Applying time-aware ranking")
        ranked_documents = self.apply_time_aware_ranking(query, documents)
        
        # Step 3: Format context
        logger.info("Formatting context")
        context = self.format_context(ranked_documents)
        
        # Step 4: Generate answer
        logger.info("Generating answer")
        answer = self.generate_answer(query, context)
        
        # Step 5: Prepare result
        sources_used = list(set(doc["source"] for doc in ranked_documents))
        
        # Log sources and their contribution
        source_counts = defaultdict(int)
        for doc in ranked_documents:
            source_counts[doc["source"]] += 1
            
        source_stats = [f"{source}: {count}/{self.final_top_k}" for source, count in source_counts.items()]
        logger.info(f"Sources used: {', '.join(source_stats)}")
        
        return {
            "query": query,
            "answer": answer,
            "documents": ranked_documents,
            "sources_used": sources_used
        }
        
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate system on test data
        
        Args:
            test_data: List of question-answer pairs
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not test_data:
            return {"error": "No test data provided"}
            
        results = []
        bleu_scores = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            question = item["question"]
            reference_answer = item["answer"]
            
            # Get generated answer
            result = self.answer_question(question)
            generated_answer = result["answer"]
            
            # Calculate BLEU score
            reference_tokens = [nltk.word_tokenize(reference_answer.lower())]
            candidate_tokens = nltk.word_tokenize(generated_answer.lower())
            
            try:
                bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=self.smoother)
            except Exception as e:
                logger.error(f"Error calculating BLEU score: {e}")
                bleu = 0.0
                
            bleu_scores.append(bleu)
            
            # Store result
            result_item = {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "bleu": bleu,
                "sources_used": result["sources_used"]
            }
            
            results.append(result_item)
            
        # Calculate metrics
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        
        # Calculate source usage statistics
        source_counts = defaultdict(int)
        for result in results:
            for source in result["sources_used"]:
                source_counts[source] += 1
                
        source_usage = {source: count / len(results) for source, count in source_counts.items()}
        
        # Compile evaluation results
        eval_results = {
            "avg_bleu": avg_bleu,
            "source_usage": source_usage,
            "results": results
        }
        
        return eval_results

def main():
    """Main function to demonstrate the system"""
    parser = argparse.ArgumentParser(description="Heterogeneous Medical QA System")
    parser.add_argument("--t5_model", type=str, default="google/flan-t5-base", 
                        help="HuggingFace T5 model name")
    parser.add_argument("--faiss_index", type=str, required=False, 
                        help="Path to FAISS index file")
    parser.add_argument("--mapping_file", type=str, required=False, 
                        help="Path to index mapping file")
    parser.add_argument("--test_file", type=str, required=False, 
                        help="Path to test data JSON file")
    parser.add_argument("--pubmed_api_key", type=str, required=False, 
                        help="NCBI API key for PubMed")
    args = parser.parse_args()
    
    # Initialize knowledge sources
    knowledge_sources = []
    
    # Add FAISS-based knowledge source if provided
    if args.faiss_index and args.mapping_file:
        faiss_source = FAISSKnowledgeSource(
            name="FAISS-Medical",
            index_path=args.faiss_index,
            mapping_file_path=args.mapping_file
        )
        knowledge_sources.append(faiss_source)
        
    # Add PubMed source
    pubmed_source = PubMedKnowledgeSource(
        name="PubMed",
        api_key=args.pubmed_api_key
    )
    knowledge_sources.append(pubmed_source)
    
    # Add clinical guidelines source
    guidelines_source = ClinicalGuidelinesKnowledgeSource(
        name="ClinicalGuidelines"
    )
    knowledge_sources.append(guidelines_source)
    
    # Add medical textbook source
    textbook_source = MedicalTextbookKnowledgeSource(
        name="MedicalTextbook"
    )
    knowledge_sources.append(textbook_source)
    
    # Initialize QA system
    qa_system = HeterogeneousMedicalQA(
        knowledge_sources=knowledge_sources,
        t5_model_name=args.t5_model
    )
    
    # Interactive mode if no test file provided
    if not args.test_file:
        print("Heterogeneous Medical QA System")
        print("Enter 'exit' to quit")
        
        while True:
            query = input("\nQuery: ")
            
            if query.lower() == "exit":
                break
                
            result = qa_system.answer_question(query)
            
            print("\nAnswer:", result["answer"])
            print("\nSources used:", ", ".join(result["sources_used"]))
            
    # Evaluation mode if test file provided
    else:
        try:
            with open(args.test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                
            eval_results = qa_system.evaluate(test_data)
            
            print(f"Average BLEU score: {eval_results['avg_bleu']:.4f}")
            print("\nSource usage statistics:")
            for source, usage in eval_results["source_usage"].items():
                print(f"  {source}: {usage:.2%}")
                
            # Save detailed results to file
            output_file = "evaluation_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2)
                
            print(f"\nDetailed results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error in evaluation mode: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()