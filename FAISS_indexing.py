import os
import json
import pickle
import time
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List, Dict

def mean_pooling(token_embeddings, attention_mask):
    """Mean pooling to create a single vector per text"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def build_faiss_index(combined_qa_file: str, output_dir: str, batch_size: int = 16):
    """Build a FAISS index using BioBERT embeddings for the combined QA dataset.
    
    Args:
        combined_qa_file (str): Path to the combined QA pairs JSON file.
        output_dir (str): Directory to save the FAISS index and embeddings.
        batch_size (int): Batch size for generating embeddings.
    """
    print("Starting indexing process...")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load combined QA dataset
    print(f"Loading combined QA dataset from {combined_qa_file}")
    with open(combined_qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    print(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Load BioBERT model and tokenizer
    # We'll use BioBERT (clinical BioBERT) which is specifically trained for biomedical text
    print("Loading BioBERT model and tokenizer...")
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Prepare for embedding generation
    # For RAG, we'll embed the combined "question + answer" text
    print("Preparing texts for embedding...")
    texts = []
    for qa_pair in qa_pairs:
        # Combine question and answer for better retrieval context
        combined_text = f"Question: {qa_pair['question']} Answer: {qa_pair['answer']}"
        texts.append(combined_text)
    
    # Generate embeddings in batches to manage memory
    print(f"Generating embeddings using batch size of {batch_size}...")
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Get model output
            outputs = model(**encoded_input)
            
            # Mean pooling
            batch_embeddings = mean_pooling(outputs.last_hidden_state, encoded_input['attention_mask'])
            embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]  # Dimension of BioBERT embeddings
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to FAISS index")
    
    # Create a mapping from index to original QA pair
    index_to_qa = {i: qa_pairs[i] for i in range(len(qa_pairs))}
    
    # Save the index and mapping
    print("Saving FAISS index and mappings...")
    index_path = os.path.join(output_dir, 'medical_qa_faiss_index.index')
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    # Save the mapping and embeddings
    mapping_path = os.path.join(output_dir, 'medical_qa_index_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump({
            'index_to_qa': index_to_qa,
            'embeddings': embeddings
        }, f)
    print(f"Saved index mapping and embeddings to {mapping_path}")
    
    print(f"Indexing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # Paths - adjust as needed
    combined_qa_file = '/scratch/lpanch2/IR_Project_final/database/train.json'
    output_dir = '/scratch/lpanch2/IR_Project_final/index'
    
    # Build the index
    build_faiss_index(combined_qa_file, output_dir)