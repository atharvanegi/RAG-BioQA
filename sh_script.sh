#!/bin/bash
#SBATCH --job-name=FAISS_monot5
#SBATCH --nodes=1
#SBATCH --partition=a100-8-gm320-c96-m1152
#SBATCH --output=new_log/%x-%j_.out
#SBATCH --error=new_log/%x-%j_.err
#SBATCH --gpus=1
#SBATCH --time=80:00:00
#SBATCH --mem=128GB

# Set environment variables for PyTorch distributed training
# export MASTER_ADDR=$(hostname -s)
# export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
# export WORLD_SIZE=2

# # Set environment variables to help with stability
# export CUDA_LAUNCH_BLOCKING=1
# export TOKENIZERS_PARALLELISM=false
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run with distributed training launcher
# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     /scratch/lpanch2/IR_Project_final/finetune_T5.py


# python /scratch/lpanch2/IR_Project_final/answer_generation.py \
#   --model_name "google/flan-t5-base" \
#   --test_data "/scratch/lpanch2/IR_Project_final/database/test.json" \
#   --faiss_index "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index" \
#   --mapping_file "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl" \
#   --output_file "new_output/unfinetuned_T5_FAISS_results.json" \
#   --batch_size 16 \
#   --top_k 3

# python /scratch/lpanch2/IR_Project_final/finetuned_T5_FAISS.py \
#   --model_dir "/scratch/lpanch2/IR_Project_final/t5_model/best_model" \
#   --test_data "/scratch/lpanch2/IR_Project_final/database/test.json" \
#   --faiss_index "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index" \
#   --mapping_file "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl" \
#   --output_file "/scratch/lpanch2/IR_Project_final/new_output/finetuned_T5_FAISS_evaluation_results.json" \
#   --batch_size 8 \
#   --top_k 4

# python /scratch/lpanch2/IR_Project_final/FAISS_colbert.py \
#   --t5_model_name "google/flan-t5-base" \
#   --colbert_model_name "colbert-ir/colbertv2.0" \
#   --test_data "/scratch/lpanch2/IR_Project_final/database/test.json" \
#   --faiss_index "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index" \
#   --mapping_file "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl" \
#   --initial_k 16 \
#   --top_k 4

# python /scratch/lpanch2/IR_Project_final/FAISS_monot5.py \
#   --t5_model_name "google/flan-t5-base" \
#   --monot5_model_name "castorini/monot5-base-msmarco" \
#   --test_data "/scratch/lpanch2/IR_Project_final/database/test.json" \
#   --faiss_index "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index" \
#   --mapping_file "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl" \
#   --initial_k 16 \
#   --top_k 4

python /scratch/lpanch2/IR_Project_final/FAISS_monot5.py --test_data "/scratch/lpanch2/IR_Project_final/database/test.json" --faiss_index "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index" --mapping_file "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl" --output_file "results/monot5_results.json" --initial_k 16 --top_k 4

# python /scratch/lpanch2/IR_Project_final/FAISS_BM25.py \
#   --t5_model_name "google/flan-t5-base" \
#   --test_data "/scratch/lpanch2/IR_Project_final/database/test.json" \
#   --faiss_index "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index" \
#   --mapping_file "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl" \
#   --initial_k 16 \
#   --top_k 4 \
#   --bm25_k1 1.5 \
#   --bm25_b 0.75

# python /scratch/lpanch2/IR_Project_final/heterogeneous_medical_qa.py \
#   --t5_model google/flan-t5-base \
#   --test_file "/scratch/lpanch2/IR_Project_final/database/test.json" \
#   --faiss_index "/scratch/lpanch2/IR_Project_final/index/medical_qa_faiss_index.index" \
#   --mapping_file "/scratch/lpanch2/IR_Project_final/index/medical_qa_index_mapping.pkl"