# RAG Multimodal Chatbot for Audio, Video & Documents - Khushmeet - Bachelor's Project

**Author:** Khushmeet
**Course:** Bachelor of Technology in Computer Science Engineering
**Date:** April 2024
**Repository Link:** https://github.com/khushmeetarora/RAG_chatbot.git

## 📄 Project Description

This project is a Streamlit web application developed as part of my final year Bachelor's work. It allows users to upload multiple audio files (e.g., MP3, WAV) and documents (PDF, DOCX, TXT) simultaneously. The application processes these files to extract or transcribe their content, embeds the text using sentence transformers, and stores the embeddings in a local FAISS vector store cache.

Users can then engage in a conversation with a Large Language Model (LLM), specifically a quantized Mistral-based model run locally using LlamaCpp. The LLM uses the combined context retrieved from *all* currently uploaded and active files (via LangChain's RAG capabilities) to answer user questions. This demonstrates the ability to synthesize information from diverse sources in a conversational manner.

## ✨ Features

*   **Multi-File Upload:** Supports uploading multiple audio, video and document files within a single session.
*   **Audio Transcription:** Utilizes OpenAI's Whisper model (via Hugging Face Transformers) for robust speech-to-text conversion.
*   **Document Text Extraction:** Handles PDF, DOCX, and TXT files using PyPDFLoader, python-docx, and textract.
*   **Content Embedding:** Generates vector embeddings for text content using Sentence Transformers (`all-MiniLM-L6-v2`).
*   **Vector Storage & Caching:** Creates and caches individual FAISS vector stores for each file to speed up reprocessing.
*   **Context Merging:** Combines the vector stores of all active files into a single context for querying.
*   **Local LLM Interaction:** Uses LlamaCpp to run a quantized LLM (e.g., Mistral 7B variant) locally for inference.
*   **Conversational Interface:** Employs LangChain (`ConversationalRetrievalChain`) and `streamlit-chat` for a user-friendly chat experience based on the uploaded content.
*   **Session Management:** Allows clearing the session to start fresh with new files.

## 🚀 Technology Stack

*   **Web Framework:** Streamlit
*   **Orchestration/RAG:** LangChain
*   **LLM Inference:** LlamaCpp (running a GGUF-quantized model, e.g., `capybarahermes-2.5-mistral-7b.Q2_K.gguf`)
*   **ASR Model:** OpenAI Whisper Large v3 (via `transformers`)
*   **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2` via `sentence-transformers`)
*   **Vector Store:** FAISS (`faiss-cpu`)
*   **Document Loaders:** `pypdf`, `python-docx`, `textract`
*   **Core Libraries:** Python 3.9, PyTorch, Transformers, Numpy

## ⚙️ Setup and Installation

**Prerequisites:**
*   Python (version 3.9 recommended)
*   Git
*   C++ Compiler (required for `llama-cpp-python` installation - see below)
*   (Optional but Recommended) Anaconda or Miniconda for environment management.

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/khushmeetarora/RAG_chatbot.git
    ```

2.  **Create and activate a virtual environment:**
    *   Using `venv`:
        ```bash
        # On Linux/macOS
        python3 -m venv venv
        source venv/bin/activate

        # On Windows (cmd)
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   Using `conda`:
        ```bash
        conda create -n chat_with_files python=3.9 # Or your preferred Python 3.x version
        conda activate chat_with_files
        ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **CRITICAL Dependency Notes:**
    *   **`llama-cpp-python`:** This package requires a C++ compiler (like GCC/Clang on Linux, Xcode Command Line Tools on macOS, or MSVC Build Tools on Windows) and CMake. Installation can take time and might fail if build tools are missing. Refer to the [official llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python) for detailed, OS-specific installation instructions if `pip install` fails. For GPU acceleration (not default in `requirements.txt`), specific compilation flags are needed (e.g., `CMAKE_ARGS="-DLLAMA_CUBLAS=on"`).
    *   **`textract`:** This package relies on external command-line tools for some file types. You might need to install them via your system's package manager. For example, on Debian/Ubuntu:
        ```bash
        sudo apt-get update && sudo apt-get install -y antiword poppler-utils tesseract-ocr libreoffice # Add others as needed by textract
        ```
        Check the `textract` documentation for requirements based on the file types you need beyond PDF/DOCX/TXT.

## 🔧 Configuration

1.  **Download the LLM Model:**
    *   You need a GGUF-quantized language model compatible with the installed `llama-cpp-python` version.
    *   A suggested model is `capybarahermes-2.5-mistral-7b.Q2_K.gguf`, which you can find on Hugging Face (search for it). Download the `.gguf` file. [Download link for the model- "https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/capybarahermes-2.5-mistral-7b.Q2_K.gguf"]
    *   Smaller quantizations (e.g., Q4_K_M) offer a better balance of performance and quality if your hardware allows.

2.  **Place the Model File:**
    *   Create a directory named `models` inside the project's root folder.
    *   Place the downloaded `.gguf` file inside this `models` directory.
    *   Alternatively, you can place the model file anywhere and set the `MODEL_DIRECTORY` environment variable to the path of the directory containing the model file before running the app.

3.  **Adjust Configuration (Optional):**
    *   Edit the `config.py` file if you need to:
        *   Change the `LLAMA_MODEL_FILENAME` if you downloaded a different model.
        *   Use a smaller Whisper model (e.g., `"openai/whisper-base"`) by changing `WHISPER_MODEL_ID` if you have limited VRAM/RAM or want faster (but less accurate) transcription.
        *   Modify LlamaCpp parameters (`LLAMA_TEMPERATURE`, `LLAMA_N_CTX`, etc.).
        *   Adjust retriever settings (`RETRIEVER_K`).

## ▶️ Running the Application

1.  Ensure your virtual environment is activated.
2.  Make sure the LLM model file is correctly placed (see Configuration).
3.  Navigate to the project's root directory in your terminal.
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
5.  The application should open automatically in your web browser.

## 🖱️ Usage

1.  Use the sidebar to upload one or more audio, video and/or document files.
2.  Wait for the application to process the files. Status updates will appear in the sidebar. Processing involves transcription/text extraction, embedding generation, and vector store creation/caching.
3.  Once processing is complete for at least one file, the "Active Files in Context" list will update, and the chat interface will become active (using the combined context).
4.  Type your questions about the content of the uploaded files into the chat input box at the bottom and press Enter or click Send.
5.  The LLM will generate a response based on the information retrieved from the relevant sections of your uploaded files.
6.  Use the "Clear Session & Start Over" button to remove all files and chat history and begin with a fresh context.

## 📂 Project Structure

├── app.py # Main Streamlit application code
├── config.py # Configuration variables (paths, model IDs, parameters)
├── requirements.txt # Python dependencies
├── README.md # This file
├── .gitignore # Specifies intentionally untracked files for Git
├── vector_store_cache/ # Directory for cached FAISS vector stores (auto-created)
├── models/ # Directory to place downloaded LLM model file (needs manual creation)
│ └── [your_model.gguf] # (Download and place the LLM GGUF file here)
└── [sample_files/] # (Optional: Use small sample files for testing)
  ├── sample.mp3
  └── sample.pdf
## ⚠️ Limitations

*   **Resource Intensive:** Running Whisper (especially larger variants) and the LLM requires significant RAM and CPU, or VRAM if using GPU acceleration. Performance heavily depends on hardware.
*   **Processing Time:** Transcription and initial file processing can be slow, especially for large audio files or documents.
*   **Accuracy:**
    *   Transcription accuracy depends on audio quality and the chosen Whisper model size.
    *   LLM responses depend on the chosen model's capabilities, quantization level, and the quality of the retrieved context. Hallucinations are possible.
    *   Text extraction quality varies by document complexity and format.
*   **Context Window:** The LLM's context window (`n_ctx` in `config.py`) limits the amount of chat history and retrieved document context it can consider simultaneously.
*   **FAISS Merging:** The approach used for merging FAISS stores might be sensitive to updates in the `faiss-cpu` library.
*   **Error Handling:** While basic error handling is implemented, complex failures in underlying libraries might require debugging.

## 📜 License

MIT License

Copyright (c) 2025 khushmeetarora

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
