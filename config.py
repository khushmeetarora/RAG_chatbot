# config.py
import os

# --- Model Paths & IDs ---
# Instructions:
# 1. Download the GGUF model file (e.g., from Hugging Face).
# 2. Place it in a 'models' subdirectory within your project, OR
# 3. Set the 'MODEL_DIRECTORY' environment variable to the path containing the model file.

LLAMA_MODEL_FILENAME = "capybarahermes-2.5-mistral-7b.Q2_K.gguf" # Or your specific GGUF file
# Define the directory where models are stored. Default is 'models' folder relative to this config file.
MODELS_DIR = os.getenv("MODEL_DIRECTORY", os.path.join(os.path.dirname(__file__), "models"))
LLAMA_MODEL_PATH = os.path.join(MODELS_DIR, LLAMA_MODEL_FILENAME)

WHISPER_MODEL_ID = "openai/whisper-large-v3" # Consider "openai/whisper-base" for easier setup/lower VRAM
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Cache Directory ---
VECTOR_STORE_CACHE_DIR = "vector_store_cache"

# --- LlamaCpp Settings ---
LLAMA_TEMPERATURE = 0.75
LLAMA_TOP_P = 1.0
LLAMA_N_CTX = 4096 # Context window size for the LLM

# --- Whisper Settings ---
WHISPER_MAX_NEW_TOKENS = 128
WHISPER_CHUNK_LENGTH_S = 30
WHISPER_BATCH_SIZE = 16 # Reduce if experiencing Out-of-Memory issues on GPU
WHISPER_RETURN_TIMESTAMPS = True
WHISPER_LANGUAGE = "english" # Set to specific language for potentially better accuracy

# --- Retriever Settings ---
RETRIEVER_K = 3 # Number of relevant document chunks to retrieve for context