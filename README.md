# README for the code structure of NegLLM


## Environment Setup

**Install Dependencies**

First, install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
**Prepare Local Llama Model**

Ensure you have downloaded and prepared the local Llama model.

**Run Local Llama Model**

We use `vllm` to serve the local model. Use the following command to start the model server:
```bash
vllm serve path/to/llama3-8b-instruct \
  --dtype auto \
  --port xxxx \
  --api-key token-abc123 \
  --gpu_memory_utilization 0.6
```
Alternatively, you can run the model using the following script:
```bash
sh 0_run_vllm.sh
```

## Step 1: Generating Negotiation Trees

The code for generating negotiation trees is in the `1_case_retrieval` folder.
Inside this folder:
- `envs` is the prototype of three negotiation scenarios. Each generation of a negotiation tree and each evaluation will create from the prototype, including randomness in creating scenarios.
- `load_scenario.py` introduce the method for create scenario from prototype.
- `short_term_strategy.py` defines the data structure of negotiation trees and nodes. 
- `step_multidimensional_action_generation.py` is the expansion step.
- `step_rollout.py` is the rollout step.

To generate negotiation trees using the Llama model via self-play, run the following command. Make sure to modify the `--llama_url` and `--llama_dir` to match your local setup.
```bash
python "1_case_retrieval/main_process.py" \
    --tree_num 5\
    --max_negotiation_round 8\
    --candidate_num 3\
    --pgmcts_iterations 10\
    --env_path "1_case_retrieval/envs" \
    --output_path "1_case_retrieval/output" \
    --llama_url http://localhost:xxxx/v1 \
    --llama_dir path/to/llama \
    --llama_api_key token-abc123
```
or
```bash
sh 1_case_retrieval.sh
```

`--tree_num` specifies the number of trees to generate for each main viewer per scene. For example, with `--tree_num=5` and three scenes, generating trees with main viewer for both sides will result in a total of 30 negotiation trees.

`--output_path` specifies the path of output negotiation trees. 
`1_case_retrieval/output_NegLLM` includes part of negotiation trees of NegLLM.
In each sample, the file name of a negotiation tree is `{scenario_name}@{main_viewer}@1.json`, and the `scenario` folder is the instantiated scenario. (mainviewer a for p1, and b for p2.)


## Step 2: Extracting Action-level Data

The code for generating negotiation trees is in the `2_extract` folder.
Inside this folder:
- `prepare_data.py` and `action_level_recog.py` includes the process from raw negotiation trees to fine-tuning data.

To extract high-quality action-level data, run the following command.
```bash
python "2_extract/prepare_data.py" \
    --data_dir 1_case_retrieval/output_NegLLM \
    --output_dir 2_extract/data
```
or
```bash
sh 2_extract.sh
```

The extracted data is saved in `--output_dir`.
Inside the output directory, there exists json files with the naming convention `{scenario_name}@{main_viewer}.json`, and a sub-folder `final` for the final data file `data_total.json` and three data file for RQ2. (mainviewer a for p1, and b for p2.)

## Step 3: Fine-tuning and Evaluation

The code for generating negotiation trees is in the `3_eval` folder.
Inside this folder, sub-folders with rq1/2/3 correspond to the experimental results of the three RQs respectively.

We use llamafactory for fine-tuning. After specifying the `model_name_or_path`, `dataset_dir`, and `dataset` in `3_0_finetune.sh`, run this command:
```bash
sh 3_0_finetune.sh
```

To run tests for two negotiation agents, first complete the API and key fields in config.py, and then run this command:
```bash
python 3_eval/main_test.py \
    --env_dir 1_case_retrieval/envs \
    --output_dir 3_eval/output \
    --temp_dir 3_eval/temp \
    --times 1
```
or
```bash
sh 3_1_test.sh
```




For statistics, run this command with `--dir` being the target folder:
```bash
python "3_eval/statistics.py" \
    --dir "3_eval/rq1_output_final"
```
or
```bash
sh 3_2_statistics.sh
```