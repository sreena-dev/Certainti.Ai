# Annotated from the typing Module

## Overview
The `Annotated` type from Python's `typing` module is useful for adding metadata to type hints. While type hints specify the expected data type, `Annotated` allows attaching extra information, which can be used for validation, constraints, documentation, or processing hints.

## Why Use `Annotated`?
- **Better Code Documentation** – Helps explain what a type represents (e.g., a 768D embedding).
- **Validation & Constraints** – Ensures values meet specific requirements (e.g., embedding length).
- **Interoperability with Libraries** – Works with `pydantic`, `FastAPI`, `dataclasses`, etc.
- **Improves Readability & Maintainability** – Developers understand the meaning of a type without checking the implementation.

## Syntax
```python
from typing import Annotated

AnnotatedType = Annotated[BaseType, metadata1, metadata2, ...]
```
### Components
- **BaseType**: The actual data type (e.g., `int`, `str`, `list[str]`).
- **metadata1, metadata2, ...**: Extra information, such as constraints, descriptions, or validation rules.

## Example Usage
### Adding Constraints for a Numeric Value
```python
from typing import Annotated
from pydantic import Field

Age = Annotated[int, Field(ge=18, le=100)]  # Age must be between 18 and 100
```

### Using `Annotated` in FastAPI
```python
from fastapi import FastAPI, Query
from typing import Annotated

app = FastAPI()

@app.get("/items/")
def read_items(limit: Annotated[int, Query(gt=0, lt=100)]):
    return {"limit": limit}
```

### Descriptive Metadata for Documentation
```python
Embedding = Annotated[list[float], "768D embedding vector"]
```

In **Retrieval-Augmented Generation (RAG) models**, `Annotated` types can be used in **embedding storage, retrieval, and validation**. The main areas where `Annotated` types are useful include:

---

## **1. Enforcing Embedding Dimension Constraints**
When working with embeddings, they have **fixed dimensions** (e.g., 768D for BERT, 1536D for OpenAI). `Annotated` helps enforce these constraints.

### **Example: Validating Embedding Size**
```python
from typing import Annotated, List

# Define an embedding type with a fixed dimension
Embedding1536D = Annotated[List[float], "Embedding vector with 1536 dimensions"]

def store_embedding(embedding: Embedding1536D):
    if len(embedding) != 1536:
        raise ValueError("Embedding must be exactly 1536 dimensions")
    print("Embedding stored successfully!")

# Example Usage
embedding = [0.1] * 1536  # ✅ Correct size
store_embedding(embedding)

# store_embedding([0.1] * 1000)  # ❌ Would raise a ValueError
```
✅ This ensures only **1536D embeddings** are passed.

---

## **2. Using `Annotated` in Pydantic for API Validation (FastAPI)**
If the **RAG model** is exposed as an API, you can use `Annotated` to enforce embedding size constraints on inputs.

### **Example: FastAPI Endpoint for Storing Embeddings**
```python
from typing import Annotated, List
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

# Define an embedding type with 768 dimensions
Embedding768D = Annotated[List[float], Field(min_items=768, max_items=768, description="768D vector")]

class EmbeddingRequest(BaseModel):
    embedding: Embedding768D
    document_id: str

@app.post("/store_embedding/")
async def store_embedding(request: EmbeddingRequest):
    return {"message": f"Embedding stored for document {request.document_id}"}

# Run the API and test via /docs or /openapi.json
```
✅ This ensures only **valid embeddings** are accepted by the API.

---

## **3. Using `Annotated` in ChromaDB Storage for RAG**
Since RAG models **store embeddings** in a vector database like **ChromaDB**, you can enforce proper embedding types when adding/querying data.

### **Example: Storing Embeddings in ChromaDB**
```python
from typing import Annotated, List
import chromadb

# Define an annotated type for embeddings
Embedding1536D = Annotated[List[float], "1536D OpenAI Embedding"]

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = chroma_client.get_or_create_collection(name="documents")

def add_to_chromadb(id: str, embedding: Embedding1536D, metadata: dict):
    if len(embedding) != 1536:
        raise ValueError("Embedding must be 1536D")
    
    collection.add(
        ids=[id],
        embeddings=[embedding],
        metadatas=[metadata]
    )
    print(f"Added document {id} to ChromaDB")

# Example Usage
embedding = [0.1] * 1536  # ✅ Correct
add_to_chromadb("doc_123", embedding, {"title": "AI Research Paper"})
```
✅ This ensures embeddings match the **model-generated dimensions**.

---

## **4. Using `Annotated` for RAG Querying**
When querying documents from the **vector database**, you need to ensure the search embedding matches the stored dimensions.

### **Example: Querying ChromaDB with Correct Embedding Size**
```python
from typing import Annotated, List

# Define an embedding type for querying
QueryEmbedding768D = Annotated[List[float], "Query vector with 768 dimensions"]

def query_rag(embedding: QueryEmbedding768D):
    if len(embedding) != 768:
        raise ValueError("Query embedding must be 768D")
    
    # Perform vector search (Example)
    print("Querying knowledge base...")

# Example Usage
query_embedding = [0.2] * 768  # ✅ Valid query embedding
query_rag(query_embedding)
```
✅ Prevents **dimension mismatches** when retrieving documents.

---

## **Why Use `Annotated` in RAG Models?**
1. **Ensures Correct Embedding Sizes** – Prevents errors from **dimension mismatches**.
2. **Improves Readability** – Clearly defines what the embedding **should be**.
3. **Works with APIs & Databases** – Helps **validate inputs** in FastAPI and **store embeddings** in ChromaDB.
4. **Prevents Runtime Errors** – Catches **wrong embedding sizes early** instead of failing in retrieval.

---

### **Final Thoughts**
If you're building a **Flask + ChromaDB** RAG system, using `Annotated` can help:
- **Store** embeddings correctly.
- **Validate** API inputs.
- **Query** with correctly-sized embeddings.
- **Ensure consistency** between model-generated embeddings and the database.

## Conclusion
The `Annotated` type enhances type hints by making them more informative and useful. It is particularly beneficial when working with frameworks like FastAPI and Pydantic, where validation and metadata play a significant role.

