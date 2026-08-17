import os
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATASET_PATH = os.path.join(DATA_DIR, "APDDataset.xlsx")
MILVUS_DB_PATH = os.path.join(DATA_DIR, "arthritis_vector_db.pkl")
COLLECTION_NAME = "arthritis_patients"

class LocalVectorDB:
    def __init__(self, path):
        self.path = path
        self.data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.data = pickle.load(f)

    def drop_collection(self, name):
        self.data = []

    def insert(self, collection_name, data):
        self.data.extend(data)
        with open(self.path, 'wb') as f:
            pickle.dump(self.data, f)
        return {"insert_count": len(data)}

    def search(self, collection_name, data, limit, output_fields):
        query_vec = np.array(data[0])
        results = []
        for item in self.data:
            vec = np.array(item["vector"])
            # Cosine similarity
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            results.append((sim, item))
        
        results.sort(key=lambda x: x[0], reverse=True)
        top_k = results[:limit]
        
        out = []
        for sim, item in top_k:
            out.append({"distance": float(sim), "entity": {k: item[k] for k in output_fields}})
        return [out]

def upload_to_vector_db():
    print("Loading data...")
    df = pd.read_excel(DATASET_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    text_data = []
    for i, row in df.iterrows():
        desc = (
            f"Patient {i}: "
            f"{'Male' if row.get('Gender_M', 0)==1 else 'Female'}, Age {row.get('Age', 'N/A')}. "
            f"ESR: {row.get('ESRh', 'N/A')} mm/hr, CRP: {row.get('CRP', 'N/A')} mg/L, "
            f"RA Factor: {row.get('RA', 'N/A')} IU/mL, Hemoglobin: {row.get('Hb', 'N/A')} g/dL."
        )
        text_data.append(desc)
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(text_data)
    
    print(f"Connecting to Local Vector DB at {MILVUS_DB_PATH} ...")
    client = LocalVectorDB(MILVUS_DB_PATH)
    client.drop_collection(COLLECTION_NAME)
    
    print(f"Uploading {len(df)} records into vector db '{COLLECTION_NAME}'...")
    insert_data = []
    for i in range(len(df)):
        insert_data.append({
            "id": i,
            "vector": embeddings[i].tolist(),
            "text": text_data[i]
        })
    
    res = client.insert(collection_name=COLLECTION_NAME, data=insert_data)
    print("Vector DB insertion successful!")
    print("Uploaded entities:", res.get("insert_count", len(insert_data)))
    
    query = "Female patient with high ESR and inflammation"
    query_emb = model.encode([query])
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_emb,
        limit=3,
        output_fields=["text"]
    )
    
    print("\\n--- Test RAG Query ---")
    print(f"Query: '{query}'")
    for hits in results:
        for hit in hits:
            print(f"Score: {hit['distance']:.4f}, Record: {hit['entity']['text']}")

if __name__ == "__main__":
    upload_to_vector_db()
