## Overview

This repository contains a documentation driven system for identifying deep learning APIs that are safe and suitable for automated test generation. The system filters APIs entirely through their documentation (`__doc__`), without executing the API. This is Stage 1 of a larger multi stage pipeline for automated LLM based test generation.

This stage produces:

* a CSV of accepted APIs (`testable_apis.csv`)
* a CSV of rejected APIs with reasons (`rejected_apis.csv`)

This ensures that unsafe, ambiguous, or non deterministic APIs do not progress to Stage 2, which generates structured JSON specifications.

---

# **1. Motivation**

Deep learning libraries expose thousands of APIs. Many cannot be fuzzed safely because they:

* require external resources
* mutate global state
* depend on IO
* have incomplete documentation
* return views or aliasing tensors
* are device specific (GPU or CUDA)
* are deprecated or unstable

A documentation first filter is essential for safety.
This repository implements that filter precisely and deterministically.

---

# **2. System Role in the Full Pipeline**

This corresponds to **Stage 1** of the global test generation system described in the project specification.
The pipeline is documentation first and only uses code when absolutely needed.

**Pipeline**

1. **Auto discovery and filtering** (this script)
2. Doc to JSON specification
3. Template and mutation rule generation
4. Automated test generation and execution

This README covers Stage 1 only.

---

# **3. Testability Rules**

My current code implements a six step decision process. Below is the complete human readable policy, the reasoning behind each rule, and examples of APIs that fail each rule.

---

## **Step 1: Documentation Availability**

A docstring must be meaningful and non empty.

Rejected when:

* documentation is missing
* documentation is too short
* placeholder text appears
* documentation is only a function signature

**Examples**

1. Missing doc
   `torch._C.ClassType` → rejected (missing doc)

2. Placeholder

```
"TODO: implement"
```

3. Signature only

```
layer(x) -> Tensor
```

Reason: no semantics available.

---

## **Step 2: Parameter Validity**

Documentation must describe parameters clearly.

Rejected when:

* no parameter section
* no required parameter
* parameters are too vague
* parameters have no type information

**Examples**

1. Missing parameter description
   `torch.distributions.utils.broadcast_all` (signature visible but no description)

2. All optional parameters
   APIs like `torch.manual_seed(seed=None)`

3. Vague types

```
x: some object
```

---

## **Step 3: Constructible Input Types**

Required parameters must have types that can be created programmatically without any external resource.

Allowed types: int, float, bool, str, tuple, list, ndarray, tensor

Rejected types: file paths, model instances, handles, connections, sessions, device objects.

**Examples**

1. Non constructible

```
path (file path)
```

2. Device specific
   `device (cuda device)` → rejection

3. Model or module
   Many `torch.nn` modules require `nn.Module` instances or graph handles.

---

## **Step 4: Side Effect Filtering**

An API is rejected if documentation mentions side effects involving:

* file IO
* GPU or hardware
* context managers
* registration or global state
* training or backward
* distributed execution
* async callbacks or hooks

**Examples**

1. IO
   `torch.save` → rejected

2. Hardware
   Any doc mentioning "GPU", "CUDA", "accelerator" → rejected

3. Training
   `optimizer.step()` → rejected (updates model parameters)

4. Distributed
   `torch.distributed.broadcast` → rejected

---

## **Step 5: Return Value Clarity**

Valid API must return something deterministic and testable.

Rejected when:

* return type is unclear
* only side effect is described
* function returns views or aliasing tensors
* in place modifications
* nondeterministic behavior

**Examples**

1. No return description

```
Performs the operation. Does not return anything.
```

2. In place operation
   `torch.add_` → rejected

3. Aliasing

```
returns a view on the same underlying storage
```

4. Mutable global state

```
updates internal state
```

---

# **4. Examples From CSV Files**

### **Accepted example (torch_accepted.csv)**

```
torch.abs
Computes the absolute value of each element in input.
Returns a tensor
```

Why accepted:

* full documentation exists
* input is a tensor
* no IO or global state
* deterministic
* clear return value

---

### **Rejected example (torch_rejected.csv)**

```
torch.sqrt
reason: unspecified failure
```

Actual failure explanation from the rules:

* return description exists
* but documentation does not include constructible input specification
* or it hits the aliasing rules or missing param types

The README will include whichever top rejected entries you want. The CSV contains hundreds; this README only includes a demonstration.

---

# **5. How to Run**

The script supports any Python library.

### Basic usage

```
python doc2info.py --lib torch
```

### Full command (recommended)

```
python doc2info.py --lib torch --out doc2info/torch_accepted.csv --rej doc2info/torch_rejected.csv --max-depth 15
```

### Parameters

| flag        | meaning                               |
| ----------- | ------------------------------------- |
| --lib       | name of the importable Python library |
| --out       | path for accepted APIs csv            |
| --rej       | path for rejected APIs csv            |
| --max-depth | recursion depth for module traversal  |

---

# **6. Output Files**

### testable_apis.csv

Contains only APIs whose docs satisfy all testability rules.

Columns:

| api_full_name | api_doc_text |

### rejected_apis.csv

Contains APIs that fail at least one rule.

Columns:

| api_full_name | api_doc_text | reason |

---

# **7. Justification of the Filtering Policy**

This policy is required because:

* Stage 2 LLM calls must not be polluted by unsafe APIs
* APIs requiring I/O or state mutation cannot be fuzzed safely
* CUDA or GPU dependent APIs are non deterministic across machines
* APIs with ambiguous or placeholder documentation will mislead an LLM
* In place or aliasing semantics break mutation based fuzzing
* Model and graph objects cannot be constructed automatically

This creates a standardized, deterministic, library agnostic filtering stage.

---