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