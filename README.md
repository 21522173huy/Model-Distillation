
# Description
- This repo implemented Model Distillation (Transfer Knowledge) Strategies
- Purpose: Observing approach that can help light-weight model outperforms LLMs in specific NLP task
- Task: Sentiment Analysis
- Dataset: `tweet_eval` with the configuration `sentiment` on HuggingFace
- Teacher Model: Llama-2-7b-hf (6.74B)
- Student Model: Roberta-Base (125M)
- Source : Knowledge Distillation Tutorial (https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)

## Setting Up
### Installtion
```
git clone -b master https://github.com/21522173huy/Model-Distillation.git
cd Model-Distillation
```
### Enviroments Variables
```
set HF_TOKEN=<your_hf_token>
set Teacher_PATH=meta-llama/Llama-2-7b-hf
set Student_PATH=FacebookAI/roberta-base
```
## Run Script
### Teacher Finetuning
```
python teacher/finetune.py
```
### Student Training
```
python student/train.py \ 
--teacher_checkpoint <your_teacher_checkpoint_folder> \
--epochs 5
```
### Evaluation
```
python evaluation.py \
--teacher_checkpoint <your_teacher_checkpoint_folder>
--student_checkpoint <your_student_checkpoint_folder>
--save_path results.json
```

### Results
|  Model | Params |Accuracy | Precision | Recall | F1 |
| -------- | ------- |------- | -------- |-------- |-------- |
| Teacher  | 6.74B |-------- |-------- |-------- |--------
| Student  |  125M |--------| --------  |-------- |-------- |


## AI used
- Name : Copilot
- Prompts: 
  - Which layers should i apply to LLama for LoRA Adaption
  - Which optimizer should i use for Finetuning phase
  - My student training phase is not good, add Scheduler.


