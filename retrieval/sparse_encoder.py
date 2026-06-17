# retrieval/sparse_encoder.py
from pinecone_text.sparse import BM25Encoder
import boto3
import os
import logging
from functools import lru_cache
import asyncio
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_bm25_from_s3():
    """
    Download and cache the BM25 model from S3.
    Uses caching to avoid re-downloading on every query.
    """
    bucket_name = os.getenv("BM25_S3_BUCKET", "echofind-models")
    key = os.getenv("BM25_S3_KEY", "models/bm25/bm25_model.pkl")
    # NEW:
    if os.name == 'nt':  # Windows
        local_path = os.path.join(os.getcwd(), "bm25_model.json")
    else:
        local_path = "/tmp/bm25_model.json"
    
    try:
        # Check if already downloaded
        if not os.path.exists(local_path):
            logger.info("Downloading BM25 model from S3...")
            
            # Try using boto3 with credentials
            try:
                from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
                
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_DEFAULT_REGION
                )
                s3_client.download_file(bucket_name, key, local_path)
                logger.info("BM25 model downloaded successfully using boto3")
                
            except Exception as boto_error:
                logger.warning(f"Failed to download with boto3: {boto_error}")
                
                # Fallback to direct HTTP if credentials not available
                import requests
                s3_url = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{key}"
                response = requests.get(s3_url, stream=True)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info("BM25 model downloaded successfully using direct HTTP")
        
        # Load using pinecone-text's BM25Encoder
        bm25 = BM25Encoder()
        bm25.load(local_path)
        
        # Verify the model loaded correctly
        params = bm25.get_params()
        doc_freq = params.get('doc_freq', {})
        
        if isinstance(doc_freq, dict):
            vocab_size = len(doc_freq.get('indices', []))
        else:
            vocab_size = 0
            
        if vocab_size == 0:
            raise ValueError("BM25 model has no vocabulary!")
            
        logger.info(f"BM25 model loaded with {vocab_size:,} vocabulary terms")
        logger.info(f"Average doc length: {params.get('avgdl', 0):.1f}")
        
        return bm25
        
    except Exception as e:
        logger.error(f"Failed to load BM25 model from S3: {e}")
        logger.warning("Falling back to default MS MARCO BM25 encoder")
        # Use pinecone-text's default MS MARCO model as fallback
        return BM25Encoder.default()

# Global instance that will be reused
_GLOBAL_BM25_ENCODER = None

def get_bm25_encoder():
    """Get or create the global BM25 encoder instance"""
    global _GLOBAL_BM25_ENCODER
    if _GLOBAL_BM25_ENCODER is None:
        _GLOBAL_BM25_ENCODER = load_bm25_from_s3()
    return _GLOBAL_BM25_ENCODER

def encode_query(text: str) -> Dict[str, Any]:
    """
    Encode a single query text into sparse vector format.
    
    Args:
        text: Query text to encode
        
    Returns:
        Dict with 'indices' and 'values' keys for Pinecone sparse vector
    """
    bm25 = get_bm25_encoder()
    
    # encode_queries returns a list, we want the first element
    sparse_vectors = bm25.encode_queries([text])
    
    if sparse_vectors and len(sparse_vectors) > 0:
        return sparse_vectors[0]
    else:
        return {'indices': [], 'values': []}

def encode_document(text: str) -> Dict[str, Any]:
    """
    Encode a single document text into sparse vector format.
    Note: Use this for indexing documents, not queries.
    
    Args:
        text: Document text to encode
        
    Returns:
        Dict with 'indices' and 'values' keys for Pinecone sparse vector
    """
    bm25 = get_bm25_encoder()
    
    # encode_documents returns a list, we want the first element
    sparse_vectors = bm25.encode_documents([text])
    
    if sparse_vectors and len(sparse_vectors) > 0:
        return sparse_vectors[0]
    else:
        return {'indices': [], 'values': []}

async def get_sparse_embeddings_batch(texts: List[str]) -> List[Dict]:
    """
    Generate sparse embeddings for a batch of texts using the pre-trained BM25 model.
    This is the async version for compatibility with existing code.
    
    Args:
        texts: List of query texts to encode
        
    Returns:
        List of sparse vector dicts
    """
    bm25 = get_bm25_encoder()
    
    # Use asyncio.to_thread for async compatibility
    # encode_queries already handles batch processing
    sparse_vectors = await asyncio.to_thread(bm25.encode_queries, texts)
    
    return sparse_vectors

def get_sparse_embeddings_batch_sync(texts: List[str]) -> List[Dict]:
    """
    Synchronous version of batch sparse embedding generation.
    
    Args:
        texts: List of query texts to encode
        
    Returns:
        List of sparse vector dicts
    """
    bm25 = get_bm25_encoder()
    return bm25.encode_queries(texts)

# Alternative: Pinecone's inference API (if you have access)
async def get_pinecone_sparse_embeddings(pinecone_client, texts: List[str]) -> List[Dict]:
    """
    Generate sparse embeddings using Pinecone's inference API.
    Requires pinecone-sparse-english-v0 model access.
    
    This is an alternative if you have access to Pinecone's inference API.
    """
    try:
        embeddings = pinecone_client.inference.embed(
            model="pinecone-sparse-english-v0",
            inputs=texts,
            parameters={"input_type": "query"}
        )
        
        sparse_vectors = []
        for emb in embeddings:
            sparse_vectors.append({
                'indices': emb.get('sparse_indices', []),
                'values': emb.get('sparse_values', [])
            })
        return sparse_vectors
        
    except Exception as e:
        logger.error(f"Failed to generate Pinecone sparse embeddings: {e}")
        # Fallback to trained BM25
        return await get_sparse_embeddings_batch(texts)

# Backward compatibility - keep the old class for any code that might use it
class BM25Encoder_DEPRECATED:
    """
    DEPRECATED: This custom implementation won't work with vectors created by pinecone-text.
    Use get_bm25_encoder() instead.
    """
    def __init__(self):
        logger.warning("Using deprecated BM25Encoder. Use get_bm25_encoder() instead.")
        self.encoder = get_bm25_encoder()
    
    def encode_query(self, text: str) -> Dict[str, Any]:
        return encode_query(text)
    
    @classmethod
    def default(cls):
        return cls()
    
    @classmethod
    def from_json(cls, json_data):
        return cls()