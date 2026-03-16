export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
python "2_extract/finetune.py" \
    --data_dir 2_extract/data \
    --output_dir 2_extract/output_model \
    --llama_dir path/to/llama \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 5e-6   
