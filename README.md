# **Deep Learning API Documentation Filter (Stage 1)**

This repository implements **Stage 1** of a multi-stage LLM-driven test-generation pipeline.
It performs **documentation-only filtering** to identify APIs that are safe, deterministic, and suitable for automated test generation.
No API is executed; decisions are made strictly from `__doc__`.

Stage 1 outputs:

* **accepted APIs** → `*_accepted.csv`
* **rejected APIs (with reasons)** → `*_rejected.csv`

These CSVs feed directly into **Stage 2: Doc→JSON specification generation**.

---

## **Pipeline Context**

1. **API Discovery + Documentation Filter (this stage)**
2. Documentation → JSON spec extraction
3. Template + mutation rule generation
4. Automated test generation + execution

---

# **Rules (Updated to Match the Code)**

Stage 1 now applies a **unified 5-step policy**, followed by a final decision layer:

### **1. Documentation Quality**

Rejected if:

* docstring missing/empty/too short
* signature-only docs
* placeholder text (`todo`, `...`, `fixme`, etc.)
* forbidden phrases (`deprecated`, `nondeterministic`, `implementation dependent`, etc.)

### **2. Parameter Validity**

Accepted if:

* function has a usable signature or structured parameter section
* at least one required parameter **or** a clear return type
  Rejected if:
* no parameter description & no valid signature
* no user-facing parameters
* required params explicitly documented with non-constructible types (files, sessions, graphs, handles, callbacks, etc.)

### **3. Constructible Types Only (Negative List)**

Required params must not depend on:

* file paths, URLs, IO streams
* sessions, graphs, handles, models
* callbacks, generators, iterators

**GPU/CUDA/device terms are now allowed**, unless tied to required parameter types.

### **4. IO-Based Side Effects**

Rejected if documentation mentions file IO:

* `read`, `write`, `save`, `load`, `open`, `download`, `serialize`, etc.

(Non-IO side-effects like GPU references or training words no longer auto-reject unless tied to type rules.)

### **5. Return Value Clarity**

Accepted if documentation shows **any** of:

* a return section
* an inline return description
* an arrow-return (`-> Tensor`, `-> float`, etc.)

Rejected only when:

* return type is completely unclear
* aliasing/view semantics (`returns a view`, `shares storage`) are present

### **6. Final Decision**

After rule evaluation:

* **TESTABLE** → included
* **NOT TESTABLE** → rejected with clear reason
* **UNCERTAIN** → rejected with best-effort explanation

---

# **Example Output**

**TORCH**
✓ 18,991 accepted
✓ 39,264 rejected
✓ 58,255 total scanned

**JAX**
✓ 485 accepted
✓ 2,673 rejected
✓ 3,158 total scanned

**TensorFlow**
✓ 9,105 accepted
✓ 27,344 rejected
✓ 36,449 total scanned

**PaddlePaddle**
✓ 6,720 accepted
✓ 35,635 rejected
✓ 42,355 total scanned

---

# **How to Run**

Minimal:

```bash
python doc2info.py --lib torch
```

Full:

```bash
python doc2info.py \
  --lib torch \
  --out torch_accepted.csv \
  --rej torch_rejected.csv \
  --max-depth 15
```

---

# **CSV Format**

### **Accepted**

`api_full_name, api_doc_text`

### **Rejected**

`api_full_name, api_doc_text, reason`

---

# **Purpose**

This filtering stage ensures that:

* Stage 2 receives only safe, deterministic APIs
* No API requiring IO or ambiguous semantics enters the LLM pipeline
* JSON specification generation remains stable
* Fuzzing avoids unsafe or impossible-to-construct inputs

This makes the entire system reproducible, documentation-aligned, and library-agnostic.


# Collect DOCS

## Pytorch
python3 doc2info/collector.py --root torch --out doc2info/torch_apis.jsonl --max-depth 5 --exclude-prefix torch.cuda --exclude-prefix torch.storage --exclude-prefix torch.sparse --exclude-prefix torch.amp --exclude-prefix torch.ao

python3 doc2info/filter_accepted.py --in doc2info/torch_apis.jsonl --outdir doc2info/results

<!-- python3 select_APIs_test.py --input results/torch/accepted.csv --output results/torch/selected.csv --target 100 -->

## Tensorflow
python3 doc2info/collector.py --root tensorflow --out doc2info/tf_apis.jsonl --max-depth 5 --exclude-prefix tensorflow.python --exclude-prefix tensorflow.dtensor.python --exclude-prefix tensorflow.compat.v1 --exclude-prefix tensorflow.compat.v2

python3 doc2info/filter_accepted.py --in doc2info/tf_apis.jsonl --outdir doc2info/results

<!-- python3 select_APIs_test.py --input results/tensorflow/accepted.csv --output results/tensorflow/selected.csv --target 100 -->

# DOCS to JSON

## PyTorch
python3 info2json/info2json.py --input doc2info/results/torch/selected.csv --outdir info2json/results/torch --model mixtral:8x7b --limit 10

## Tensorflow
python3 info2json/info2json.py --input doc2info/results/tensorflow/selected.csv --outdir info2json/results/tensorflow --model mixtral:8x7b --limit 10

# Validate JSON 
## PyTorch
python3 json_validator/json_validator.py --spec-dir info2json/results/torch --api-csv doc2info/results/torch/accepted.csv --state-dir json_validator/results/torch --repair-model mixtral:8x7b

./json_validator/results/torch/repair_retry.sh

## Tensorflow
python3 json_validator/json_validator.py --spec-dir info2json/results/tensorflow --api-csv doc2info/results/tensorflow/accepted.csv --state-dir json_validator/results/tensorflow --repair-model mixtral:8x7b

./json_validator/results/tensorflow/repair_retry.sh


# INIT

## PyTorch
python3 json2init/json2init.py --spec-dir info2json/results/torch --outdir json2init/results/torch --ok-csv json_validator/results/torch/ok.csv --smoke-test --overwrite --non-strict-smoke

python3 json2init/json2init_validator.py --spec-dir info2json/results/torch --api-csv doc2info/results/torch/accepted.csv --stage3-issues json2init/results/torch/issues.jsonl --outdir json2init/results/torch --ok-csv json_validator/results/torch/ok.csv --repair-model mixtral:8x7b --max-rounds 3

## Tensorflow
python3 json2init/json2init.py --spec-dir info2json/results/tensorflow --outdir json2init/results/tensorflow --ok-csv json_validator/results/tensorflow/ok.csv --smoke-test --overwrite --non-strict-smoke

python3 json2init/json2init_validator.py --spec-dir info2json/results/tensorflow --api-csv doc2info/results/tensorflow/accepted.csv --stage3-issues json2init/results/tensorflow/issues.jsonl --outdir json2init/results/tensorflow --ok-csv json_validator/results/tensorflow/ok.csv --repair-model mixtral:8x7b --max-rounds 3


<!-- 
# JSON to INIT

## PyTorch
python3 "json2init/json2init.py" --input "info2json/results/torch" --output "json2init/results/torch" 

## Tensorflow
python3 "json2init/json2init.py" --input "info2json/results/tensorflow" --output "json2init/results/tensorflow"

# INIT to TEST

# PyTorch
python3 "init2test/init2test.py" --spec-input "info2json/results/torch" --init-input "json2init/results/torch" --output "init2test/results/torch" --overwrite --execute

# Tensorflow
python3 "init2test/init2test.py" --spec-input "info2json/results/tensorflow" --init-input "json2init/results/tensorflow" --output "init2test/results/tensorflow" --overwrite --execute 
-->


python3 -c "import csv,json,pathlib; allow={x.strip() for x in open('doc2info/results/torch/manual_allowlist.txt',encoding='utf-8') if x.strip()}; src='doc2info/torch_apis.jsonl'; dst='doc2info/results/torch/accepted.csv'; rows=list(csv.DictReader(open(dst,encoding='utf-8'))) if pathlib.Path(dst).exists() else []; seen={r.get('api_full_name','').strip() for r in rows}; add=[]; [add.append({'api_full_name':e.get('api','').strip(),'api_doc_text':e.get('doc','')}) for e in map(json.loads, open(src,encoding='utf-8')) if e.get('api','').strip() in allow and e.get('api','').strip() not in seen]; f=open(dst,'w',encoding='utf-8',newline=''); w=csv.DictWriter(f,fieldnames=['api_full_name','api_doc_text']); w.writeheader(); w.writerows(rows+add); f.close(); print(f'added {len(add)} APIs to {dst}')" 



----

DOCKER (JSON-2-INIT)

docker build -t deepfuzz-pytorch -f docker_setup/Dockerfile.pytorch docker_setup

docker run --rm -it deepfuzz-pytorch bash -lc "python3 -c 'import torch; print(torch.__version__); print(torch.__file__)'"

docker run --rm -it -v "$PWD":/workspace/DeepFuzz -w /workspace/DeepFuzz deepfuzz-pytorch bash -lc 'python3 json2init/json2init.py --spec-dir info2json/results/torch --outdir json2init/results/torch --ok-csv json_validator/results/torch/ok.csv --smoke-test --overwrite --smoke-timeout-sec 30 --non-strict-smoke'

docker run --rm -it -v "$PWD":/workspace/DeepFuzz -w /workspace/DeepFuzz deepfuzz-pytorch bash -lc 'python3 json2init/json2init_validator.py --spec-dir info2json/results/torch --api-csv doc2info/results/torch/accepted.csv --stage3-issues json2init/results/torch/issues.jsonl --outdir json2init/results/torch --ok-csv json_validator/results/torch/ok.csv --primary-repair-model mistral:7b --fallback-repair-model mixtral:8x7b --fallback-after-round 3 --max-rounds 3 --smoke-timeout-sec 30 --repair-host http://host.docker.internal:11434 --only-stage3-retry'

docker run --rm -it -v "$PWD":/workspace/DeepFuzz -w /workspace/DeepFuzz deepfuzz-pytorch bash -lc 'python3 json2init/json2init.py --spec-dir info2json/results/torch --outdir json2init/results/torch --ok-csv json_validator/results/torch/ok.csv --smoke-test --overwrite --smoke-timeout-sec 30 --non-strict-smoke --only-api-list json2init/results/torch/retry_api_list.txt'