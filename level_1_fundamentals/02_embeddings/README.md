# Project 2: Embeddings & Semantic Search

## What You'll Learn
- What embeddings are and how they work
- Converting text to vector representations
- Calculating semantic similarity
- Building search without a vector database
- Difference between keyword and semantic search

## Key Concepts

### What are Embeddings?
Embeddings are numerical representations (vectors) of text that capture semantic meaning. Similar concepts have similar vectors.

Example:
- "headache" and "migraine" → very similar vectors (high cosine similarity)
- "headache" and "diabetes" → different vectors (low cosine similarity)

### Embedding Models
- **text-embedding-3-small**: 1536 dimensions, $0.02/1M tokens (recommended)
- **text-embedding-3-large**: 3072 dimensions, $0.13/1M tokens (higher quality)

### Cosine Similarity
Measures how similar two vectors are:
- 1.0 = identical
- 0.8-0.9 = very similar
- 0.5-0.7 = somewhat related
- 0.0 = unrelated
- -1.0 = opposite (rare in practice)

## Running the Code

```bash
python main.py
```

## How It Works

1. **Convert text to embeddings**: Each text becomes a vector of numbers
2. **Calculate similarity**: Use cosine similarity to compare vectors
3. **Rank results**: Sort by similarity score

## Exercises

### Exercise 1: Drug Interaction Search
Create a system to find similar drug interactions:
```python
drug_interactions = [
    "Warfarin and aspirin increase bleeding risk",
    "ACE inhibitors and NSAIDs may cause kidney damage",
    # Add more...
]
query = "blood thinner safety"
# Find similar interactions
```

### Exercise 2: Medical Abbreviation Expander
Build a search for medical abbreviations:
- Query: "heart rate monitoring"
- Should find: "ECG continuous cardiac monitoring"

### Exercise 3: Symptom Checker
Create a symptom similarity system:
- Input: Patient's symptom description
- Output: Most similar known symptom patterns
- Use: Help with differential diagnosis

### Exercise 4: Cost Comparison
Compare costs of embedding vs using full LLM:
- Calculate cost of embedding 10,000 medical terms
- Compare to cost of 10,000 LLM calls for similarity

## Healthcare Applications

### Clinical Decision Support
```python
# Find similar past cases
new_case = "70yo male, SOB, leg swelling, orthopnea"
similar_cases = find_most_similar(new_case, historical_cases)
```

### Medical Literature Search
```python
# Search research papers by meaning, not keywords
query = "treatments for treatment-resistant depression"
# Finds papers about ECT, TMS, ketamine, etc.
```

### ICD-10 Code Suggestion
```python
clinical_note = "Patient presents with acute bronchitis..."
similar_codes = find_most_similar(clinical_note, icd10_descriptions)
# Suggests: J20.9 (Acute bronchitis, unspecified)
```

## Important Notes

### Embeddings vs Vector Databases
This project calculates similarities in memory. For production:
- Use vector databases (ChromaDB, Pinecone, FAISS) for large datasets
- This is covered in Level 2!

### Embedding Quality Tips
1. **Clean text first**: Remove special characters, normalize whitespace
2. **Batch requests**: Embed multiple texts at once for efficiency
3. **Cache embeddings**: Don't re-embed the same text
4. **Choose right model**: Small model is good for most use cases

## Expected Output

```
🔍 Query: 'possible heart attack symptoms'
------------------------------------------------------------

1. Similarity: 0.872
   55-year-old male with crushing chest pain, pain in jaw, 
   extreme anxiety. Heavy smoker.

2. Similarity: 0.845
   45-year-old male with chest pain radiating to left arm, 
   shortness of breath, and sweating. History of hypertension.

3. Similarity: 0.798
   67-year-old female with gradual onset of chest discomfort, 
   nausea, and unusual fatigue. Diabetic.
```

## Common Issues

### Issue: Low similarity scores for obviously similar texts
**Solution**: Text preprocessing matters. Clean and normalize input text

### Issue: High API costs
**Solution**: Batch embed texts, cache results, use smaller model

### Issue: All similarities are high (>0.9)
**Solution**: Texts might be too similar or too short. Add more variety

## Performance Tips

```python
# ❌ Slow - individual requests
for text in texts:
    embedding = get_embedding(text)

# ✅ Fast - batch request
embeddings = client.embeddings.create(input=texts, model="...")
```

## Next Steps
Now that you understand embeddings, move to **03_function_calling** to learn how LLMs can call your code!
