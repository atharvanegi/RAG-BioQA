import os
import json
import pickle
import torch
import numpy as np
import faiss
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import random
from tqdm.auto import tqdm
from typing import List, Dict

# Configuration
class Config:
    # Paths
    train_file = "/scratch/lpanch2/IR_Project_final/database/train.json"
    valid_file = "/scratch/lpanch2/IR_Project_final/database/val.json"
    faiss_index = "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index"
    mapping_file = "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl"
    output_dir = "/scratch/lpanch2/IR_Project_final/t5_model"
    
    # Model settings
    model_name = "google/flan-t5-base"  # You can also try flan-t5-large if you have the resources
    max_input_length = 512
    max_output_length = 128
    
    # Training settings
    batch_size = 8  # Further reduced to avoid numerical instability
    learning_rate = 5e-5  # Lower learning rate to prevent NaN gradients
    weight_decay = 0.01
    num_train_epochs = 3
    max_grad_norm = 1.0  # Add gradient clipping
    
    # LoRA settings (Parameter-Efficient Fine-Tuning)
    lora_r = 16  # Rank of LoRA update matrices
    lora_alpha = 32  # Scaling factor for LoRA
    lora_dropout = 0.1
    
    # Retrieval settings
    num_retrieved = 4  # Number of similar QA pairs to retrieve for context
    
    # FP16 mixed precision
    use_fp16 = True
    
    # Debug mode - when enabled, uses only a small subset of data
    debug_mode = False
    debug_sample_percentage = 0.1  # 10% of data for testing


def load_data_and_index(config):
    """Load training/validation data and FAISS index"""
    print("Loading data and index...")
    
    # Load training and validation data
    with open(config.train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(config.valid_file, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
    
    # In debug mode, use only a fraction of the data
    if config.debug_mode:
        print(f"Debug mode enabled: Using {config.debug_sample_percentage * 100}% of data")
        # Set random seed for reproducibility
        random.seed(42)
        
        # Sample the data
        train_size = max(1, int(len(train_data) * config.debug_sample_percentage))
        valid_size = max(1, int(len(valid_data) * config.debug_sample_percentage))
        
        train_data = random.sample(train_data, train_size)
        valid_data = random.sample(valid_data, valid_size)
        
        print(f"Sampled {len(train_data)} training examples and {len(valid_data)} validation examples")
    
    # Load FAISS index and mapping
    index = faiss.read_index(config.faiss_index)
    
    with open(config.mapping_file, 'rb') as f:
        mapping_data = pickle.load(f)
    
    index_to_qa = mapping_data['index_to_qa']
    embeddings = mapping_data['embeddings']
    
    # Debugging: Check the structure of index_to_qa keys
    key_types = set(type(k) for k in list(index_to_qa.keys())[:10])
    print(f"Key types in index_to_qa: {key_types}")
    print(f"First few keys in index_to_qa: {list(index_to_qa.keys())[:5]}")
    
    print(f"Loaded {len(train_data)} training examples and {len(valid_data)} validation examples")
    print(f"FAISS index contains {index.ntotal} vectors of dimension {index.d}")
    
    # Fix index_to_qa if needed - ensure all keys are strings
    if int in key_types:
        print("Converting integer keys to strings in index_to_qa mapping...")
        index_to_qa = {str(k): v for k, v in index_to_qa.items()}
    
    return train_data, valid_data, index, index_to_qa, embeddings


def get_biobert_embedding(question, tokenizer, model, device):
    """Generate BioBERT embedding for a question"""
    # Tokenize input
    inputs = tokenizer(
        question, 
        max_length=512, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    
    # Get embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        token_embeddings = outputs.last_hidden_state
        embedding = torch.sum(token_embeddings * attention_mask, 1) / torch.sum(attention_mask, 1)
    
    # Convert to numpy and normalize
    embedding = embedding.cpu().numpy()
    faiss.normalize_L2(embedding)
    
    return embedding

def adjust_context_for_length(retrieved_pairs, question, tokenizer, max_length):
    """Adjust the number of retrieved pairs to fit within max token length"""
    # Start with the question itself to ensure it's always included
    question_prompt = f"Question: {question} Answer:"
    question_tokens = len(tokenizer.encode(question_prompt))
    remaining_tokens = max_length - question_tokens - 20  # 20 tokens buffer
    
    # Calculate how many pairs we can include
    usable_pairs = []
    tokens_used = 0
    context_prefix = "Context: "
    prefix_tokens = len(tokenizer.encode(context_prefix)) - 2  # -2 for special tokens
    
    for pair in retrieved_pairs:
        pair_text = f"Question: {pair['question']} Answer: {pair['answer']} "
        pair_tokens = len(tokenizer.encode(pair_text)) - 2  # -2 for special tokens
        
        if tokens_used + pair_tokens <= remaining_tokens:
            usable_pairs.append(pair)
            tokens_used += pair_tokens
        else:
            break  # Stop adding pairs
    
    return usable_pairs

def retrieve_similar_qa_pairs(question, index, index_to_qa, embeddings, query_embedding, k=4, exclude_self=True):
    """Retrieve top-k similar QA pairs for a given question"""
    # Search the FAISS index - only get k+1 to be more efficient
    D, I = index.search(query_embedding, k+1 if exclude_self else k)
    
    # Get the retrieved QA pairs
    retrieved_pairs = []
    for i in I[0]:
        # Convert index to the appropriate format for lookup
        idx_key = str(i)
        
        # Try different key formats if the direct string conversion doesn't work
        if idx_key not in index_to_qa:
            # Check if the mapping uses integers instead of strings
            if i in index_to_qa:
                idx_key = i
            else:
                # Print some debug info about the first few keys in index_to_qa
                print(f"Debug - FAISS returned index {i}, keys in mapping:", list(index_to_qa.keys())[:5])
                continue  # Skip this result if we can't find a matching key
        
        # Get the QA pair and check if it matches the query
        qa_pair = index_to_qa[idx_key]
        if exclude_self and qa_pair['question'].strip() == question.strip():
            continue
            
        retrieved_pairs.append(qa_pair)
        if len(retrieved_pairs) == k:
            break
    
    # If we couldn't find enough pairs, return what we have
    return retrieved_pairs


def prepare_training_data(train_data, valid_data, index, index_to_qa, embeddings, biobert_tokenizer, biobert_model, t5_tokenizer, config, device):
    """Prepare training and validation data with retrieved contexts"""
    print("Preparing training data with retrieved contexts...")
    
    # Function to prepare a single example
    def prepare_example(example, is_training=True):
        question = example['question']
        answer = example['answer']
        
        # Generate embedding for the question
        query_embedding = get_biobert_embedding(question, biobert_tokenizer, biobert_model, device)
        
        # Retrieve similar QA pairs
        retrieved_pairs = retrieve_similar_qa_pairs(
            question, index, index_to_qa, embeddings, query_embedding, 
            k=config.num_retrieved, exclude_self=is_training
        )
        
        # Adjust number of retrieved pairs to fit context length
        adjusted_pairs = adjust_context_for_length(
            retrieved_pairs, question, t5_tokenizer, config.max_input_length
        )
        
        # Format context string
        context = "Context: "
        for pair in adjusted_pairs:
            context += f"Question: {pair['question']} Answer: {pair['answer']} "
        
        # Create input and target strings
        input_text = f"{context}Question: {question} Answer:"
        target_text = answer
        
        return {
            "input_text": input_text,
            "target_text": target_text,
            "question": question,  # Keep original for evaluation
        }
    
    # Process training data
    processed_train = [prepare_example(example, is_training=True) for example in tqdm(train_data, desc="Processing training data")]
    
    # Process validation data
    processed_valid = [prepare_example(example, is_training=False) for example in tqdm(valid_data, desc="Processing validation data")]
    
    return processed_train, processed_valid


def tokenize_data(processed_train, processed_valid, tokenizer, config):
    """Tokenize the processed data for T5"""
    print("Tokenizing data...")
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=config.max_input_length,
            padding="max_length",
            truncation=True,
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"],
                max_length=config.max_output_length,
                padding="max_length",
                truncation=True,
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Convert to datasets format
    train_dataset = Dataset.from_dict({
        "input_text": [example["input_text"] for example in processed_train],
        "target_text": [example["target_text"] for example in processed_train],
        "question": [example["question"] for example in processed_train],
    })
    
    valid_dataset = Dataset.from_dict({
        "input_text": [example["input_text"] for example in processed_valid],
        "target_text": [example["target_text"] for example in processed_valid],
        "question": [example["question"] for example in processed_valid],
    })
    
    # Tokenize
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "target_text", "question"],
    )
    
    tokenized_valid = valid_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "target_text", "question"],
    )
    
    return tokenized_train, tokenized_valid


def setup_peft_model(config):
    """Set up the T5 model with LoRA for parameter-efficient fine-tuning"""
    print(f"Loading {config.model_name} and setting up LoRA...")
    
    # Load base model with double precision to avoid numerical instability
    model = T5ForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32  # Use full precision rather than default half precision
    )
    
    # Initialize input embeddings with small variance to prevent NaN issues
    # This helps stabilize training for transformer models
    for name, param in model.named_parameters():
        if 'embed' in name:
            print(f"Initializing {name} with normal distribution")
            torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q", "v"],  # T5's attention matrices
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Verify model parameters don't contain NaN values
    for name, param in model.named_parameters():
        if param.requires_grad and torch.isnan(param).any():
            print(f"WARNING: NaN detected in trainable parameter {name}")
    
    return model


def train_model(model, tokenizer, tokenized_train, tokenized_valid, config):
    """Train the model with Trainer"""
    print("Setting up training...")
    
    # Create a directory for output
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",  # Updated from evaluation_strategy
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # Disable mixed precision to avoid numerical instability
        max_grad_norm=config.max_grad_norm,  # Add gradient clipping
        # Distributed training settings
        ddp_find_unused_parameters=False,
        # Disable DataLoader multi-processing in favor of SLURM allocation
        dataloader_num_workers=2,  # Reduced as suggested by warning
        # Optimizer settings
        optim="adamw_torch",  # Use PyTorch's AdamW implementation
        warmup_ratio=0.1,  # Add warmup to stabilize training
    )
    
    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the best model
    trainer.save_model(os.path.join(config.output_dir, "best_model"))
    print(f"Model saved to {os.path.join(config.output_dir, 'best_model')}")


def main():
    # Initialize config
    config = Config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data and index
    train_data, valid_data, index, index_to_qa, embeddings = load_data_and_index(config)
    
    # Load BioBERT model for embedding generation
    from transformers import AutoTokenizer, AutoModel
    print("Loading BioBERT model...")
    biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(device)
    
    # Load T5 tokenizer
    print("Loading T5 tokenizer...")
    t5_tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    
    # Prepare data with retrieved contexts
    processed_train, processed_valid = prepare_training_data(
        train_data, valid_data, index, index_to_qa, embeddings, 
        biobert_tokenizer, biobert_model, t5_tokenizer, config, device
    )
    
    # Tokenize data
    tokenized_train, tokenized_valid = tokenize_data(
        processed_train, processed_valid, t5_tokenizer, config
    )
    
    # Setup model with LoRA
    model = setup_peft_model(config)
    model = model.to(device)  # Move model to GPU
    
    # Train the model using Trainer
    train_model(model, t5_tokenizer, tokenized_train, tokenized_valid, config)


if __name__ == "__main__":
    main()