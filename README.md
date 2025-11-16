# **File Overview**

### **1. `tffuzz/doc_collect.py`**

**Purpose:**
Collects raw API metadata directly from Python libraries.

**What it does:**

* Resolves `qualname` into the actual Python callable
* Extracts signature + docstring
* Detects pseudo-enums
* Returns minimal structured info (`ParamInfo`, name, module, signature)

**Used when:**
You want the **raw source metadata** for each API.

---

### **2. `tffuzz/doc_build.py`**

**Purpose:**
Builds the **final enriched API JSON spec** using LLM + fallback logic.

**What it does:**

* Calls `doc_collect` to get signature + docstring
* Prompts an LLM to produce structured param specs
* Cleans hallucinations & invalid params
* Normalizes dtypes, shapes, ranks
* Produces final `"params"` + `"oracles"` object
* Works across TensorFlow, PyTorch, JAX, NumPy, OpenCV

**Used when:**
You want the **final JSON-ready API spec** (actual output used by your testing system).

---

### **3. `tffuzz/registry.py`**

**Purpose:**
Centralised **rules + normalization logic**.

**What it does:**

* Unifies dtypes across frameworks (TF, Torch, NumPy, OpenCV)
* Normalizes dtypes from docstrings
* Applies parameter-type heuristics:
  `tensor | shape | number | bool | str | any`
* Loads user-defined override rules

**Used when:**
You want consistent **dtype/type parsing** across all libraries.

---

# **How to Run**

### **1. Collect raw API list**

Example for TensorFlow:

```bash
python tffuzz/doc_collect.py --modules tensorflow --limit 200 --out TF_API_min.json
```

Example for PyTorch:

```bash
python tffuzz/doc_collect.py --modules torch --limit 200 --out TORCH_API_min.json
```

---

### **2. Build enriched JSON spec**

```bash
python -m tffuzz.doc_build --in tffuzz/API/TORCH_API_min.json --out tffuzz/API/TORCH_API_built.json --limit 10
```

Run this **after** collecting the raw list.

---

# **Output Structure**

Example final spec object:

```json
{
  "name": "abs",
  "qualname": "tensorflow.abs",
  "params": [
    {
      "name": "x",
      "type": "tensor",
      "optional": false,
      "allowed_dtypes": ["float32", "float64", "complex64", "int32"],
      "description": "A tensor."
    }
  ],
  "oracles": { "invariants": [] }
}
```

Final output file is an array of such objects:

```
[
  { ...spec1... },
  { ...spec2... },
  ...
]
```

Use this for:

* LLM fuzzing
* Semantic validation
* Differential testing
* API compatibility analysis

---

# **Next Steps**

1. Add **JAX**, **OpenCV**, **NumPy**, **Torchvision**, **SciPy** API lists
2. Run `doc_build.py` on them
3. Train/test the fuzzer using your enriched API JSON
4. Plug into your semantic-testing pipeline

---

# **Dependencies**

### Base

```
numpy>=1.26
tensorflow>=2.14
pytest>=8
requests>=2.31
```

### PyTorch (CPU)

```
torch --index-url https://download.pytorch.org/whl/cpu
torchvision --index-url https://download.pytorch.org/whl/cpu
torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---