import os
import json
import pickle
import faiss
import numpy as np
from typing import List, Dict
import time

def combine_databases_with_embeddings(medquad_pickle_path: str, meddialog_pickle_path: str, output_dir: str) -> None:
    """Combine MedQuAD and MedDialog databases with embeddings into a single database.
    
    Args:
        medquad_pickle_path (str): Path to the MedQuAD vector database pickle file with embeddings.
        meddialog_pickle_path (str): Path to the MedDialog vector database pickle file with embeddings.
        output_dir (str): Directory to save the combined QA pairs and FAISS index.
    """
    print("Starting database combination process...")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MedQuAD data with embeddings
    print(f"Loading MedQuAD dataset from {medquad_pickle_path}")
    with open(medquad_pickle_path, 'rb') as f:
        medquad_data = pickle.load(f)
    medquad_qa_pairs = medquad_data['qa_pairs']
    medquad_vector_db = medquad_data['vector_db']
    print(f"Loaded {len(medquad_qa_pairs)} QA pairs from MedQuAD")
    print(f"MedQuAD vector DB has {medquad_vector_db.ntotal} vectors")
    
    # Load MedDialog data with embeddings
    print(f"Loading MedDialog dataset from {meddialog_pickle_path}")
    with open(meddialog_pickle_path, 'rb') as f:
        meddialog_data = pickle.load(f)
    meddialog_qa_pairs = meddialog_data['qa_pairs']
    meddialog_vector_db = meddialog_data['vector_db']
    print(f"Loaded {len(meddialog_qa_pairs)} QA pairs from MedDialog")
    print(f"MedDialog vector DB has {meddialog_vector_db.ntotal} vectors")
    
    # Verify dimensions match
    medquad_dim = medquad_vector_db.d
    meddialog_dim = meddialog_vector_db.d
    if medquad_dim != meddialog_dim:
        raise ValueError(f"Embedding dimensions don't match: MedQuAD={medquad_dim}, MedDialog={meddialog_dim}")
    else:
        dimension = medquad_dim
        print(f"Both datasets use {dimension}-dimensional embeddings")
    
    # Extract vectors from both indices
    print("Extracting vectors from FAISS indices...")
    medquad_vectors = faiss.extract_index_vectors(medquad_vector_db)[1]
    meddialog_vectors = faiss.extract_index_vectors(meddialog_vector_db)[1]
    
    # Verify vector counts match QA pair counts
    if len(medquad_qa_pairs) != medquad_vector_db.ntotal:
        print(f"Warning: MedQuAD QA pairs ({len(medquad_qa_pairs)}) don't match vector count ({medquad_vector_db.ntotal})")
    if len(meddialog_qa_pairs) != meddialog_vector_db.ntotal:
        print(f"Warning: MedDialog QA pairs ({len(meddialog_qa_pairs)}) don't match vector count ({meddialog_vector_db.ntotal})")
    
    # Combine QA pairs and remove source and url fields
    print("Combining QA pairs and preparing vectors...")
    combined_qa_pairs = []
    combined_vectors = []
    
    # Process MedQuAD pairs
    for i, pair in enumerate(medquad_qa_pairs):
        if i < medquad_vector_db.ntotal:  # Ensure we have a vector for this pair
            if pair['answer'] and pair['answer'].strip():  # Check for non-empty answer
                combined_qa_pairs.append({
                    'question': pair['question'],
                    'answer': pair['answer']
                })
                combined_vectors.append(medquad_vectors[i])
    
    # Process MedDialog pairs
    for i, pair in enumerate(meddialog_qa_pairs):
        if i < meddialog_vector_db.ntotal:  # Ensure we have a vector for this pair
            if pair['answer'] and pair['answer'].strip():  # Check for non-empty answer
                combined_qa_pairs.append({
                    'question': pair['question'],
                    'answer': pair['answer']
                })
                combined_vectors.append(meddialog_vectors[i])
    
    # Count how many empty answers were removed
    total_original = len(medquad_qa_pairs) + len(meddialog_qa_pairs)
    empty_answers_count = total_original - len(combined_qa_pairs)
    print(f"Total original QA pairs: {total_original}")
    print(f"QA pairs removed due to empty answers: {empty_answers_count}")
    print(f"QA pairs after removing empty answers: {len(combined_qa_pairs)}")
    
    # Remove duplicates based on question text
    print("Removing duplicate questions...")
    unique_qa_pairs = []
    unique_vectors = []
    seen_questions = set()
    
    for i, pair in enumerate(combined_qa_pairs):
        if pair['question'] not in seen_questions:
            seen_questions.add(pair['question'])
            unique_qa_pairs.append(pair)
            unique_vectors.append(combined_vectors[i])
    
    duplicates_removed = len(combined_qa_pairs) - len(unique_qa_pairs)
    print(f"Duplicate questions removed: {duplicates_removed}")
    print(f"Total unique QA pairs after cleaning: {len(unique_qa_pairs)}")
    
    # Convert to numpy array for FAISS
    unique_vectors = np.array(unique_vectors, dtype=np.float32)
    
    # Create new FAISS index and add vectors
    print("Creating new FAISS index...")
    combined_index = faiss.IndexFlatL2(dimension)
    combined_index.add(unique_vectors)
    print(f"Added {combined_index.ntotal} vectors to new FAISS index")
    
    # Save combined QA pairs to JSON
    print("Saving combined QA pairs to JSON...")
    combined_json_path = os.path.join(output_dir, 'combined_cleaned_qa_pairs.json')
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(unique_qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"Combined QA pairs saved to {combined_json_path}")
    
    # Save combined QA pairs and FAISS index to pickle
    print("Saving combined vector database...")
    combined_pickle_path = os.path.join(output_dir, 'combined_vector_db_with_embeddings.pkl')
    with open(combined_pickle_path, 'wb') as f:
        pickle.dump({'qa_pairs': unique_qa_pairs, 'vector_db': combined_index}, f)
    print(f"Combined vector database with embeddings saved to {combined_pickle_path}")
    
    total_time = time.time() - start_time
    print(f"Database combination completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    # Paths to the original JSON files - adjust these paths as needed
    medquad_json_path = '/scratch/lpanch2/IR_project/datasets/MedQuAD/medquad_vector_db_with_embeddings.pkl'
    meddialog_json_path = '/scratch/lpanch2/IR_project/datasets/MedDialog/meddialog_vector_db_with_embeddings.pkl'
    
    # Output directory - adjust as needed
    output_dir = '/scratch/lpanch2/IR_Project_final/database'
    
    # Combine the databases
    combine_databases_with_embeddings(medquad_json_path, meddialog_json_path, output_dir)