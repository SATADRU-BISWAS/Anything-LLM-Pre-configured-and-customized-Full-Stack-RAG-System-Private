#<img width="640" height="427" alt="AnythingLLM_DockerHub" src="https://github.com/user-attachments/assets/8874a7e5-f063-4a09-be14-db29502c6fea" />
 
# Internal RAG Pipeline with Distributed Colab Preprocessing and AnythingLLM

A production-lean deployment that ingests internal PDFs, denoises them at scale in Google Colab, and automatically embeds them into AnythingLLM (Docker) using Mistral 7B, LanceDB, and Nomic Embed Text v1. The pipeline supports GUI and REST-based ingestion, fault routing for large files, and is designed for continuous embedding automation. [1][2][3]

## Architecture overview

- Source PDFs are uploaded from client machines to Google Drive via a PowerShell uploader.  
- Cleaning and OCR-safe denoising run in parallel across 4 Colab notebooks to accelerate 7k+ document throughput.  
- Cleaned PDFs are auto-dumped to a designated client folder, then pushed to AnythingLLM via REST for vectorization using LanceDB and Nomic Embed Text v1. [3][4][5]
- Large-file failures are routed to a local “manual-review” folder for GUI-based ingestion.  
- Core RAG served by AnythingLLM (Docker) with Mistral 7B as the LLM. [2][6]

## Key components

- AnythingLLM (Docker) for RAG and workspace management. [2][1]
- Mistral 7B for responses; LanceDB as the vector database; Nomic Embed Text v1 for embeddings. [6][4][5]
- PowerShell uploader for Drive; Google Colab for distributed cleaning. [3]

## Images

- System diagram: Place a high-level architecture diagram image here (e.g., /docs/architecture.png).  
- Colab worker: Place a screenshot illustrating the batched Colab runtime (e.g., /docs/colab_batch.png).  
- AnythingLLM workspace: Place a screenshot of the workspace config (e.g., /docs/anythingllm_workspace.png). [3]

## Features

- RAG for internal employees with GUI and REST ingestion. [3]
- Distributed preprocessing across 4 Colab notebooks to accelerate >7k docs.  
- Automated REST push to AnythingLLM; fallback manual embedding for large files. [3]
- Dockerized deployment for easy backup, updates, and multi-platform support. [2][1]

## Prerequisites

- Docker and Docker Compose for AnythingLLM. [2]
- Google account with Drive access for staging PDFs.  
- Google Colab for preprocessing notebooks.  
- API key and base URL for AnythingLLM REST API. [7][3]
- Optional: Ollama or remote inference endpoint if running Mistral 7B locally. [8][6]

***

## RAG system setup

AnythingLLM was selected for its all-in-one RAG capabilities, flexible LLM/vector integrations, Agents, and straightforward Docker deployment for organizations. It supports workspace management, API access, and GUI upload out of the box. [1][3]

- Docker quickstart (example docker-compose.yml):
```yaml
version: '3.8'
services:
  anythingllm:
    image: mintplexlabs/anythingllm:latest
    container_name: anythingllm
    ports:
      - "3001:3001"
    environment:
      - JWT_SECRET=change_me
      - STORAGE_DIR=/app/storage
      - VECTOR_DB=lance
      - EMBEDDING_MODEL=nomic-embed-text-v1
      - LLM_PROVIDER=ollama
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - DEFAULT_LLM=mistral
    volumes:
      - ./storage:/app/storage
      - ./data:/app/data
    restart: unless-stopped
```
This sample shows LanceDB as the vector store, Nomic-Embed-Text for embeddings, and Mistral (via Ollama) as the LLM. Adjust to your infra. [2][4][5][8][3]

***

## PowerShell uploader to Google Drive

A PowerShell script uploads source PDFs from a local source folder to a Drive folder, enabling Colab access. This approach minimizes friction for non-technical employees.

- Requirements:
  - A Drive folder with a known Folder ID.
  - OAuth or Service Account flow via Google APIs (or rclone for simplicity).

- Example using rclone backend:
```powershell
param(
  [Parameter(Mandatory=$true)][string]$SourcePath,
  [Parameter(Mandatory=$true)][string]$RemoteName,       # e.g., "gdrive:"
  [Parameter(Mandatory=$true)][string]$RemoteFolderPath  # e.g., "company-ingest/raw"
)

# Ensure rclone is installed and 'RemoteName' is configured.
# Upload PDFs only
Get-ChildItem -Path $SourcePath -Filter *.pdf -Recurse | ForEach-Object {
    $relative = $_.FullName.Substring($SourcePath.Length).TrimStart('\')
    $dest = Join-Path $RemoteFolderPath $relative
    $dest = $dest -replace '\\','/'

    Write-Host "Uploading $($_.FullName) -> $RemoteName$dest"
    rclone copy $_.FullName "$RemoteName$dest" --create-empty-src-dirs --progress --transfers=8 --checkers=16 --drive-chunk-size=128M
}
```
This script pushes PDFs into a consistent Drive hierarchy for Colab workers. [3]

***

## PDF denoising and cleaning in Colab


---

📄 Batch PDF Processing Pipeline with OCR, LangChain & ChromaDB

This implements a production-grade batch PDF ingestion pipeline designed for RAG (Retrieval-Augmented Generation) systems. It processes multiple PDFs from a Google Drive folder, automatically applies OCR fallback, performs chunking, hashing, vectorization, and finally stores processed files into structured Google Drive directories.


---



🚀 Key Features

🔍 Batch PDF Detection (auto scans Drive folder for new PDFs)

🧾 Metadata Extraction + OCR Fallback for scanned/unreadable PDFs

🧩 Text Chunking optimized for retrieval

🔐 Chunk Hashing to ensure deduplication

🧠 HuggingFace Embeddings integrated via LangChain

🗂️ Persistent ChromaDB Vector Store

📁 Google Drive File Management

Moves processed PDFs → ProcessedDocs/

Moves failed PDFs → FailedDocs/


🔄 Fully automated end-to-end pipeline



---

```

📦 Architecture Overview

Google Drive (Input Folder)
        │
        ├── Batch Loader (glob / Drive API)
        │
        ├── Document Processor
        │     ├── PDF Text Extraction
        │     ├── Metadata Check
        │     ├── OCR Fallback if Needed (Tesseract)
        │     └── Chunking + Hashing
        │
        ├── Embedding Generator (HuggingFace → LangChain)
        │
        ├── Vector Storage (ChromaDB)
        │
        └── Google Drive Output Folder
              ├── ProcessedDocs/
              └── FailedDocs/
```

---

📁 Google Drive Folder Structure

```

MyDrive/
 ├── InsuranceDocs/        # Input PDFs - the batch source
 ├── ProcessedDocs/        # PDFs successfully processed and indexed
 └── FailedDocs/           # PDFs that failed OCR or text extraction

```
---

⚙️ How the Pipeline Works

1️⃣ Batch PDF Detection

The pipeline detects all PDFs inside the Google Drive input folder:

pdf_files = glob.glob("/content/drive/MyDrive/InsuranceDocs/*.pdf")

This enables large-scale automated ingestion — no manual file selection required.


---

2️⃣ Per-File Processing Loop

Each PDF is processed sequentially:

for pdf_path in pdf_files:
    process_document(pdf_path)

This ensures uniform handling across clean PDFs, scanned PDFs, and mixed-content documents.


---

3️⃣ Metadata Extraction + OCR Fallback

The system first attempts normal PDF text extraction:

doc = PyPDFLoader(pdf_path).load()

If metadata or text extraction fails (common in scanned PDFs), the system automatically applies OCR:

text = run_tesseract_ocr(pdf_path)

OCR fallback triggers when:

Extracted text is empty

Critical metadata is missing

Extraction errors occur

Pages contain only images



---

4️⃣ Chunking & Hashing

Extracted text is segmented into retrieval-friendly chunks:

chunks = chunk_document(text)

Each chunk is hashed for deduplication:

chunk_id = sha256(chunk_text)

Why hashing?

Prevents duplicate data in the vector store

Detects repeated documents across multiple uploads

Ensures ChromaDB inserts are idempotent



---

5️⃣ Embeddings + ChromaDB Insert

Text chunks are embedded using HuggingFace models (via LangChain) and stored in ChromaDB:

store.add_texts(
    texts=[chunk.page_content],
    metadatas=[chunk.metadata],
    ids=[chunk_id]
)

Using chunk hashes as IDs effectively eliminates duplicates.


---

6️⃣ Storing Processed PDFs in Google Drive

After successful vectorization, the file is moved to the processed folder:

shutil.move(pdf_path, "/content/drive/MyDrive/ProcessedDocs/")

This ensures:

No file is processed twice

Input folder stays clean

Easy audit trail of processed documents



---

7️⃣ Handling Failures

If extraction or OCR fails:

shutil.move(pdf_path, "/content/drive/MyDrive/FailedDocs/")

Keeps faulty uploads isolated from the main ingestion flow.


---

🧠 End-to-End Summary

1. Load all PDFs from Drive batch folder


2. Loop through each file


3. Extract metadata → OCR fallback if required


4. Chunk + hash text for deduplication


5. Generate embeddings with HuggingFace


6. Store chunks in ChromaDB


7. Move processed PDFs to ProcessedDocs/


8. Move failures to FailedDocs/



This pipeline is designed for reliability, automation, and scalability for any RAG-based document intelligence system.




---

🛠️ Tech Stack

Component	Technology

OCR	Tesseract OCR
LLM Pipeline	LangChain
Embeddings	HuggingFace Transformers
Vector DB	ChromaDB (Persistent Mode)
File Storage	Google Drive
PDF Parsing	PyPDF2 / PyPDFLoader
Hashing	SHA256



---

📌 Ideal Use Cases

Insurance document indexing

Contract repositories

Legal PDF knowledge bases

Financial statement ingestion

Enterprise RAG systems

OCR-heavy enterprise workflows



```

# -*- coding: utf-8 -*-
"""Batch 3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1og69qR0MiruiRslztal2zDSW_iFli3ww
"""

from google.colab import files
uploaded = files.upload()

# FULL OCR PIPELINE - integrated: auth, batch download, image processing, run pipeline
# Run this in Colab after uploading client_secrets.json or placing it in your Satadru Drive.

# --------------------------- INSTALL & IMPORTS --------------------------- #
!pip install -U -q PyDrive pdf2image opencv-python-headless pandas imutils img2pdf pytesseract
!apt-get install -y -qq tesseract-ocr poppler-utils

import os, json, gc, zipfile, traceback, datetime, time, shutil
import numpy as np
import pandas as pd
import cv2, imutils
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import drive as colab_drive, files as colab_files

# --------------------------- AUTHENTICATION --------------------------- #
# 1) Reliance Drive via PyDrive (client_secrets.json must be present in runtime or in Satadru Drive)
gauth = GoogleAuth()
gauth.LoadClientConfigFile("client_secrets.json")   # ensure file is present
gauth.CommandLineAuth()  # copy-paste auth code when prompted
rel_drive = GoogleDrive(gauth)

# 2) Mount Satadru personal Drive (satadru.ola@gmail.com) to persist files
colab_drive.mount('/content/my_drive', force_remount=True)

# --------------------------- PATH CONFIG & CREATION --------------------------- #
SATADRU_BASE = "/content/my_drive/MyDrive/OCR_Pipeline"
LOGS_DIR = os.path.join(SATADRU_BASE, "Logs")
INPUT_DIR = os.path.join(SATADRU_BASE, "Input_PDFs")          # persisted downloaded PDFs
OUTPUT_FOLDER = os.path.join(SATADRU_BASE, "OCR_Output")      # âœ… permanent folder for processed PDFs
TEMP_IMAGE_FOLDER = "/content/temp_images"                    # ephemeral (fast) temporary images

METRICS_CSV = os.path.join(LOGS_DIR, "pdf_quality_metrics.csv")
TRACKER_FILE = os.path.join(LOGS_DIR, "progress_tracker.json")
BATCH_FILE_IN_SATADRU = os.path.join(LOGS_DIR, "batch_3.json")  # persisted copy of batch JSON (if present)
ERROR_LOG_FILE = os.path.join(LOGS_DIR, "error_log.txt")

DPI = 200
ZIP_BATCH_SIZE = 500   # how many PDFs per ZIP before saving/uploading/downloading

# ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)     # âœ… now inside Satadru Drive for persistence
os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)

print("Paths ready:")
print(" - Logs:", LOGS_DIR)
print(" - Input PDFs (persisted):", INPUT_DIR)
print(" - Output (persisted):", OUTPUT_FOLDER)
print(" - Temp images (ephemeral):", TEMP_IMAGE_FOLDER)


# --------------------------- TRACKER HELPERS --------------------------- #
def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {"processed": []}

def save_tracker(tracker):
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)

def log_error(pdf_path, page_num, error):
    with open(ERROR_LOG_FILE, "a") as f:
        f.write(f"---\nPDF: {pdf_path}, Page: {page_num}\n")
        f.write(traceback.format_exc())
        f.write("\n")

# --------------------------- IMAGE PROCESSING HELPERS --------------------------- #
def denoise_image(image):
    """Reduce speckle and compression noise."""
    try:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    except Exception:
        return image

def orientation_fix(image):
    """Auto-rotate using OCR-based orientation detection with a fallback."""
    try:
        osd = pytesseract.image_to_osd(image)
        rotation = int([line for line in osd.split("\n") if "Rotate" in line][0].split(":")[-1].strip())
        if rotation != 0:
            image = imutils.rotate_bound(image, -rotation)
    except Exception:
        try:
            text_normal = pytesseract.image_to_string(image)
            flipped = cv2.rotate(image, cv2.ROTATE_180)
            text_flipped = pytesseract.image_to_string(flipped)
            if len(text_flipped.strip()) > len(text_normal.strip()):
                image = flipped
        except Exception:
            pass
    return image

def preprocess_image(image):
    """Downscale huge images, denoise and fix orientation. Returns (denoised, fixed)."""
    try:
        if image.shape[0] > 8000 or image.shape[1] > 8000:
            scale = 8000.0 / max(image.shape[0], image.shape[1])
            new_w = int(image.shape[1] * scale)
            new_h = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"âš ï¸ Downscaled large image to {image.shape}")
    except Exception as e:
        print(f"âš ï¸ Warning in resizing: {e}")
    denoised = denoise_image(image)
    fixed = orientation_fix(denoised)
    return denoised, fixed

def save_pdf(image_list, output_pdf_path):
    """Combine processed images back into a single PDF."""
    if not image_list:
        return
    images = [Image.fromarray(img).convert("RGB") for img in image_list]
    images[0].save(output_pdf_path, save_all=True, append_images=images[1:])
    print(f"[OK] Saved OCR-ready PDF: {output_pdf_path}")

# --------------------------- MAIN PROCESS FUNCTION --------------------------- #
def analyze_and_preprocess_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    preprocessed_images = []
    char_counts, blur_list, contrast_list, failed_pages = [], [], [], []
    metrics = {"PDF_File": pdf_name}

    try:
        info = pdfinfo_from_path(pdf_path)
        num_pages = info.get('Pages', None)
    except Exception:
        num_pages = None

    try:
        if num_pages is None:
            images = convert_from_path(pdf_path, dpi=DPI)
        else:
            images = [convert_from_path(pdf_path, dpi=DPI, first_page=i, last_page=i)[0]
                      for i in range(1, num_pages + 1)]
    except Exception as e:
        log_error(pdf_path, "ALL", e)
        metrics.update({"Status": "Failed", "Failed_Pages": "ALL"})
        return metrics, None

    for page_num, image in enumerate(images, start=1):
        try:
            np_img = np.array(image)
            denoised, final_img = preprocess_image(np_img)
            preprocessed_images.append(final_img)

            gray = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
            blur_list.append(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
            contrast_list.append(np.std(gray))
            text = pytesseract.image_to_string(final_img)
            char_counts.append(len(text.strip()))

            # cleanup per-page
            del np_img, denoised, final_img, gray, text
            gc.collect()
        except Exception as e:
            failed_pages.append(page_num)
            log_error(pdf_path, page_num, e)

    metrics.update({
        "Num_Pages": len(images),
        "Avg_Blur": float(np.mean(blur_list)) if blur_list else 0.0,
        "Avg_Contrast": float(np.mean(contrast_list)) if contrast_list else 0.0,
        "Avg_CharCount": float(np.mean(char_counts)) if char_counts else 0.0,
        "Failed_Pages": ",".join(map(str, failed_pages)) if failed_pages else "",
        "Status": "Success" if (len(failed_pages) < len(images)) else "Failed"
    })

    if metrics["Avg_CharCount"] == 0 and metrics["Status"] != "Failed":
        metrics["Status"] = "Failed"
        metrics["Failed_Pages"] = "ALL"

    return metrics, preprocessed_images

# --------------------------- BATCH DETECTION & DOWNLOAD --------------------------- #
# Decide whether to use existing tracker or download batch_3.json from Reliance Drive
if os.path.exists(TRACKER_FILE):
    print("âœ… Tracker found in Satadru Drive - resuming.")
    tracker = load_tracker()
    processed = set(tracker.get("processed", []))
    # Load batch file if present to know assigned PDF set
    if os.path.exists(BATCH_FILE_IN_SATADRU):
        with open(BATCH_FILE_IN_SATADRU, "r") as f:
            batch_files = json.load(f)
    else:
        # fallback: treat all files in INPUT_DIR as the batch (if any)
        batch_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")])
else:
    print("âš ï¸ No tracker found. Attempting to fetch batch_3.json from Reliance Drive.")
    file_list = rel_drive.ListFile({'q': "title='batch_3.json' and trashed=false"}).GetList()
    if not file_list:
        raise FileNotFoundError("batch_3.json not found in Reliance Drive. Upload it into Reliance Logs.")
    file_list[0].GetContentFile(BATCH_FILE_IN_SATADRU)
    with open(BATCH_FILE_IN_SATADRU, "r") as f:
        batch_files = json.load(f)
    tracker = {"processed": []}
    processed = set()

print(f"Assigned batch: {len(batch_files)} PDFs")
# Compute remaining PDFs to download/process
remaining_pdfs = [p for p in batch_files if os.path.basename(p) not in processed]

# Download only remaining PDFs from Reliance Drive into SATADRU INPUT_DIR
for pdf_name in remaining_pdfs:
    local_path = os.path.join(INPUT_DIR, pdf_name)
    if os.path.exists(local_path):
        print(f"â†³ already exists locally: {pdf_name}")
        continue

    # Escape single quotes for safe Drive query
    safe_name = os.path.basename(pdf_name).replace("'", "\\'")
    q = f"title contains '{os.path.splitext(os.path.basename(pdf_name))[0]}' and trashed=false"

    try:
        found = rel_drive.ListFile({'q': q}).GetList()
    except Exception as e:
        print(f"âŒ Query failed for {pdf_name}: {e}")
        with open(ERROR_LOG_FILE, "a") as elog:
            elog.write(f"Query failed for {pdf_name}: {traceback.format_exc()}\n")
        continue

    if not found:
        print(f"âš ï¸ Not found in Reliance Drive: {pdf_name}")
        with open(ERROR_LOG_FILE, "a") as elog:
            elog.write(f"Missing in Reliance drive: {pdf_name}\n")
        continue

    try:
        found[0].GetContentFile(local_path)
        print(f"âœ… Downloaded: {pdf_name}")
    except Exception as e:
        print(f"âŒ Failed to download {pdf_name}: {e}")
        with open(ERROR_LOG_FILE, "a") as elog:
            elog.write(f"Download failed: {pdf_name}\n{traceback.format_exc()}\n")

# Ensure tracker file exists
if not os.path.exists(TRACKER_FILE):
    save_tracker(tracker)

# --------------------------- RUN PIPELINE --------------------------- #
PDF_FOLDER_LOCAL = INPUT_DIR
tracker = load_tracker()
metrics_df = pd.read_csv(METRICS_CSV) if os.path.exists(METRICS_CSV) else pd.DataFrame()

# Build list of PDFs to process (preserve ordering from batch_files)
all_pdfs = [os.path.join(PDF_FOLDER_LOCAL, f) for f in batch_files if os.path.basename(f).lower().endswith(".pdf")]
total_pdfs = len(all_pdfs)
print(f"Starting processing loop: {total_pdfs} total (will skip already-processed).")

batch_counter = 0
for pdf_path in all_pdfs:
    file_name = os.path.basename(pdf_path)
    tracker = load_tracker()   # reload to minimize concurrent write surprises
    if file_name in tracker.get("processed", []):
        # skip already processed
        continue

    print(f"\nâž¡ï¸ Processing: {file_name}")
    try:
        metrics, preprocessed_images = analyze_and_preprocess_pdf(pdf_path)
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

        if preprocessed_images:
            output_pdf_path = os.path.join(OUTPUT_FOLDER, file_name)
            save_pdf(preprocessed_images, output_pdf_path)
            batch_counter += 1

        # update tracker & metrics persistently
        tracker = load_tracker()
        tracker.setdefault("processed", []).append(file_name)
        save_tracker(tracker)
        metrics_df.to_csv(METRICS_CSV, index=False)
        print(f"âœ… Progress: {len(tracker['processed'])}/{total_pdfs} PDFs processed")

    except Exception as e:
        print(f"âŒ Fatal error while processing {file_name}: {e}")
        log_error(pdf_path, "ALL", e)
        # don't append to processed, so it can be retried later

    # ---- ZIP AND SAVE AFTER EACH BATCH ----
    if batch_counter >= ZIP_BATCH_SIZE or pdf_path == all_pdfs[-1]:
        zip_name = f"/content/OCR_Batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files_in_folder in os.walk(OUTPUT_FOLDER):
                for file in files_in_folder:
                    zipf.write(os.path.join(root, file), arcname=file)
        print(f"ðŸ“¦ Created ZIP archive: {zip_name}")
        # If you want the browser to download automatically, uncomment the next line:
        # colab_files.download(zip_name)

        # OPTIONAL: upload ZIP back to Reliance Drive or Satadru Drive (uncomment & edit target folder)
        # upload_file = rel_drive.CreateFile({'title': os.path.basename(zip_name), 'parents': [{'id': '<RELIANCE_TARGET_FOLDER_ID>'}]})
        # upload_file.SetContentFile(zip_name)
        # upload_file.Upload()
        # print(f"â¬†ï¸ Uploaded ZIP to Reliance Drive: {upload_file['title']}")

        # cleanup ephemeral output folder for next batch
        # cleanup only temp images (not Drive outputs)
        try:
            shutil.rmtree(TEMP_IMAGE_FOLDER)
            os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)
        except Exception as e:
            print(f"âš ï¸ Temp cleanup failed: {e}")
        batch_counter = 0
        print("ðŸ§¹ Cleaned temp images after batch ZIP")



print("\nðŸŽ¯ Pipeline run complete. Tracker and metrics saved in Satadru Drive Logs.")
print(f" - Tracker: {TRACKER_FILE}")
print(f" - Metrics CSV: {METRICS_CSV}")
print(f" - Error log: {ERROR_LOG_FILE}")

```

---


***

## Decentralized processing with 4 Colab notebooks with dividing the unprocessed pds into separate batches.

---

📄 OCR Preprocessing & Batch Processing Pipeline

This repository contains a complete OCR preprocessing workflow designed for large-scale PDF processing.
It supports:

Page-level image preprocessing (denoise, auto-orientation, downscale)

OCR quality analysis (blur, contrast, character count)

PDF-to-image conversion using Poppler

Tesseract-based orientation detection and OCR scoring

Automatic batching, zipping, and cleanup

Distributed multi-account/multi-worker processing with batch JSON files

Persistent logging (progress tracker, error logs, quality metrics)


This pipeline is optimized for long-running Google Colab sessions with Google Drive persistence.


---

📁 Directory Structure


```
OCR_Pipeline/
│
├── ocr_script.ipynb / ocr_script.py       # Main pipeline
├── Logs/
│   ├── progress_tracker.json              # Tracks processed files
│   ├── error_log.txt                      # Detailed error traces
│   └── pdf_quality_metrics.csv            # OCR quality metrics
└── batch_splitter.py                      # Multi-worker batch generator


```

---

🚀 Main OCR Pipeline

Below is the full annotated code used for:

PDF → image conversion

Per-page preprocessing

OCR scoring

Saving processed PDFs

Batch zipping & cleanup

Progress tracking



---

📦 Install & Setup


```
!apt-get install poppler-utils
!pip install pdf2image==1.16.3 pytesseract numpy pandas pillow opencv-python matplotlib scikit-image

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')

```
---

⚙️ Configuration

```
import os, gc, json, time, numpy as np, pandas as pd, traceback
import cv2
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
from PIL import Image
import imutils
import matplotlib.pyplot as plt
from google.colab import files

PDF_FOLDER = "/content/drive/MyDrive"
TEMP_IMAGE_FOLDER = "/content/temp_images"
OUTPUT_FOLDER = "/content/OCR_Output"
LOGS_DIR = "/content/drive/MyDrive/OCR_Pipeline/Logs"

os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TRACKER_FILE = os.path.join(LOGS_DIR, "progress_tracker.json")
ERROR_LOG_FILE = os.path.join(LOGS_DIR, "error_log.txt")
METRICS_CSV = os.path.join(LOGS_DIR, "pdf_quality_metrics.csv")

DPI = 200
ZIP_BATCH_SIZE = 500

```
---

📘 Progress Tracker & Logging

```
def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {"processed": []}

def save_tracker(data):
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)

def log_error(pdf_path, page_num, error):
    with open(ERROR_LOG_FILE, "a") as f:
        f.write(f"\n\n--- ERROR in {pdf_path} PAGE {page_num} ---\n")
        f.write(error)
        f.write("\n------------------------------\n")

```
---

🖼 Image Preprocessing Helpers

```

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def orientation_fix(image):
    try:
        osd = pytesseract.image_to_osd(image)
        angle = int([x for x in osd.split() if x.isdigit()][0])
        return imutils.rotate_bound(image, angle)
    except:
        txt_normal = len(pytesseract.image_to_string(image).strip())
        flipped = cv2.rotate(image, cv2.ROTATE_180)
        txt_flipped = len(pytesseract.image_to_string(flipped).strip())
        return flipped if txt_flipped > txt_normal else image

def preprocess_image(image):
    if max(image.shape) > 8000:
        scale = 8000 / max(image.shape)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    denoised = denoise_image(image)
    oriented = orientation_fix(denoised)
    return denoised, oriented

def save_pdf(images, output_pdf):
    pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    pil_images[0].save(output_pdf, save_all=True, append_images=pil_images[1:])

```
---

🔬 Core Processing Function

```
def analyze_and_preprocess_pdf(pdf_path):
    metrics = {"PDF": pdf_path, "Status": "Success", "Failed_Pages": ""}

    try:
        images = convert_from_path(pdf_path, dpi=DPI)
    except:
        metrics["Status"] = "Failed"
        return metrics, None

    preprocessed = []
    blur_vals, contrast_vals, char_vals = [], [], []
    failed_pages = []

    for i, page in enumerate(images, 1):
        try:
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            _, processed = preprocess_image(img)

            preprocessed.append(processed)

            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            blur_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            contrast_vals.append(np.std(gray))
            char_vals.append(len(pytesseract.image_to_string(processed).strip()))
        except Exception as e:
            failed_pages.append(i)
            log_error(pdf_path, i, traceback.format_exc())

    metrics.update({
        "Num_Pages": len(images),
        "Avg_Blur": np.mean(blur_vals) if blur_vals else 0,
        "Avg_Contrast": np.mean(contrast_vals) if contrast_vals else 0,
        "Avg_CharCount": np.mean(char_vals) if char_vals else 0,
        "Failed_Pages": failed_pages,
        "Status": "Failed" if failed_pages else "Success"
    })

    return metrics, preprocessed
```

---

▶ Main Execution Loop


```
tracker = load_tracker()
metrics_df = pd.read_csv(METRICS_CSV) if os.path.exists(METRICS_CSV) else pd.DataFrame()

all_pdfs = [
    os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER)
    if f.lower().endswith(".pdf")
]

batch_counter = 0

for pdf_path in all_pdfs:
    file_name = os.path.basename(pdf_path)

    if file_name in tracker["processed"]:
        continue

    metrics, images = analyze_and_preprocess_pdf(pdf_path)
    metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

    if images:
        save_pdf(images, os.path.join(OUTPUT_FOLDER, file_name))
        batch_counter += 1

    tracker["processed"].append(file_name)
    save_tracker(tracker)
    metrics_df.to_csv(METRICS_CSV, index=False)

    if batch_counter >= ZIP_BATCH_SIZE or pdf_path == all_pdfs[-1]:
        zip_name = f"OCR_Batch_{int(time.time())}.zip"
        os.system(f"zip -r {zip_name} {OUTPUT_FOLDER}")
        files.download(zip_name)

        time.sleep(np.random.randint(900, 1200))
        os.system(f"rm -rf {OUTPUT_FOLDER}")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        batch_counter = 0


```
---

🔁 Multi-Worker Batch Splitter

Use this script to divide unprocessed PDFs across multiple accounts.
```

import os, json, math

PDF_FOLDER = "/content/drive/MyDrive"
LOGS_DIR = "/content/drive/MyDrive/OCR_Pipeline/Logs"
TRACKER_FILE = os.path.join(LOGS_DIR, "progress_tracker.json")

with open(TRACKER_FILE, "r") as f:
    tracker = json.load(f)

processed = set(tracker.get("processed", []))
all_pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
remaining = [f for f in all_pdfs if f not in processed]

num_accounts = 4
batch_size = math.ceil(len(remaining) / num_accounts)

batches = [
    remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)
]

for i, batch in enumerate(batches, 1):
    with open(os.path.join(LOGS_DIR, f"batch_{i}.json"), "w") as f:
        json.dump(batch, f, indent=2)


```
***

## Automated file upload to Anything LLM using REST API from designated client document folder
---
AnythingLLM PDF Watcher – RAM-Aware, Auto-Recovery, Fault-Tolerant Uploader

This  provides a production-grade file watcher that automatically uploads PDFs into AnythingLLM using its API.
It includes RAM-aware throttling, Docker container health checks, auto-restart, batch processing, and a progress bar — ensuring reliability even with thousands of documents.

Originally built to safely process large volumes of PDFs without crashing the AnythingLLM Docker container.

---

🚀 Features

✅ Smart Uploading

Automatically detects new PDFs in a folder

Uploads them to a specified AnythingLLM workspace

Never uploads the same file twice

Logs upload history in embedded_log.json


🧠 RAM-Aware Throttling

Pauses all uploads if container RAM ≥ 80%

Resumes only when RAM ≤ 65%

Prevents memory pressure and unexpected crashes


🛠 Container Auto-Healing

Detects when the AnythingLLM container is down

Automatically restarts it (twice if needed)

Waits for stabilization

Continues uploads without losing progress


📦 Batch Processing

Splits PDFs into size-based batches

Ideal for huge collections (e.g., 500–20,000+ PDFs)


📊 Clean Progress Display

Shows

Current batch

PDF index

RAM percentage


Real-time progress bar like:
►▸░░░░░░ 35%


🧱 Fault Tolerance

Upload attempts auto-retry (3×)

Failed PDFs moved to error_jobs/

Continues even if individual files fail



---

📁 Project Structure

 ```
anythingllm-watcher/
│
├── watcher.py                  # Main watcher script (RAM-aware + restart-safe)
├── watcher_config.json         # Configuration file (paths, API, workspace)
├── embedded_log.json           # Tracks uploaded PDFs
├── requirements.txt            # Python dependencies
│
├── error_jobs/                 # PDFs that failed after retries
│
├── samples/                    # Sample config + sample PDF
│   ├── sample.pdf
│   └── sample_config.json
│
└── docs/                       # Architectural diagrams, notes
    ├── architecture.png
    └── flow_diagram.md

```

---

🔧 Installation

1️⃣ Install Python packages

```
pip install -r requirements.txt

```

Your requirements.txt should contain:

requests

2️⃣ Configure AnythingLLM settings

Edit watcher_config.json:
```

{
  "watch_folder": "C:/Users/biswa/Downloads/Jobs",
  "anythingllm_url": "http://localhost:3001",
  "api_key": "YOUR_API_KEY",
  "workspace": "reliance",
  "max_batch_mb": 20,
  "check_interval_sec": 60
}

```

3️⃣ Enable Docker API (required for RAM monitoring)
    Enabling Docker RAM Monitoring (Required)

    Your watcher uses the Docker Engine API to read memory usage:

     http://localhost:2375/containers/anythingllm/stats?stream=false

     To enable this, you must activate Docker’s TCP port (2375).

     ✔ Activate Docker Engine API (GUI — Recommended)

1. Open Docker Desktop


2. Go to Settings → General


3. Scroll down to:

☑ Expose daemon on tcp://localhost:2375 without TLS


4. Enable it


5. Click Apply & Restart


✔ Verify:

Restart Docker Desktop.

Test:

Invoke-RestMethod http://localhost:2375/version

If you see JSON output → RAM monitoring is working.   




---

▶️ Running the Watcher


1.Navigate to the designated folder that contains python script below:

python watcher.py

2. Then simply execute the script "python watcher.py" by entering "python watcher.py" and pressing Enter "⏎" on keyboard

You’ll see output like:

👀 Watching folder: C:\Users\biswa\Downloads\Jobs

🔍 Found 615 new PDF(s).
📦 Total batches to process: 32

📦 Processing batch 1/32 (20 PDF(s))...
➡️ [1/20] ATC-1 Scanned.pdf
🧠 RAM: 32.1%
⬆️ Uploading...
✔ Uploaded OK


---

📉 What Happens During High RAM Usage?

If RAM hits ≥ 80%, you’ll see:

⛔ RAM too high (82.4%). Pausing until ≤ 65%...
🔍 RAM check: 78%
🔍 RAM check: 69%
🔍 RAM check: 64%
✅ RAM safe again. Resuming uploads.


---

🩹 Container Crash Recovery

If AnythingLLM container crashes:

⛔ Container is DOWN. Attempting restart…
♻️ Restart 1/2...
✔ Restarted successfully
⏳ Waiting 30s before RAM check...
🧠 RAM: 55% → Safe to resume.

If restart fails twice → script continues checking every 60 seconds.


---

🔥 Upload Failures

If upload fails 3×:

File moves to error_jobs/

Logged as:


"FAILED - 2025-02-03 06:12:45"


---

📝 Logging

✔ Successful upload

Added to embedded_log.json:

"C:\\path\\to\\file.pdf" : "2025-02-03 12:33:12"

✔ Failed upload

Stored as:

"C:\\path\\to\\file.pdf" : "FAILED-2025-02-03 12:33:12"


---

---


***

## AnythingLLM configuration (Docker)

The Docker edition was chosen for multi-platform integration, easier backups, and streamlined updates. It also aligns with enterprise deployment practices while preserving the desktop GUI option. [2][1]

- Core capabilities relevant here:
  - Multiple LLM providers and vector DBs.  
  - Workspaces for doc scoping and permissions.  
  - Full REST API for programmatic embedding and chat. [3][7]

Place a screenshot of the workspace and API key setup here (e.g., /docs/anythingllm_api.png). [3]



***

## Model and vector settings

- LLM: Mistral 7B for efficient, high-quality responses with strong benchmark performance and Apache 2.0 licensing. [6][9][10]
- Vector DB: LanceDB for local, high-performance vector storage compatible with modern embedding flows. [4]
- Embeddings: Nomic Embed Text v1 (and v1.5 option) supporting search_document and long-text handling parameters. [5][11][12]

If running via Ollama, ensure mistral and nomic-embed-text models are available to the embedding and inference layers. [8][4]

***

## 1. REST ingestion to AnythingLLM

Clean PDFs in the client folder are tunneled to AnythingLLM via REST for vector embedding, automating ingestion from the client machine into the target workspace. AnythingLLM exposes an API for managing workspaces and documents programmatically. [3][7]

- Example Python ingestion client:
```python
import os
import time
import requests

ANY_BASE = os.getenv("ANY_BASE_URL", "http://localhost:3001/api")
ANY_KEY  = os.getenv("ANY_API_KEY", "replace_me")
WORKSPACE_ID = os.getenv("ANY_WORKSPACE_ID", "internal-knowledge")
CLEAN_DIR = os.getenv("CLEAN_DIR", r"D:\RAG\clean")
HEADERS = {"Authorization": f"Bearer {ANY_KEY}"}

def upload_doc(path):
    url = f"{ANY_BASE}/workspaces/{WORKSPACE_ID}/documents"
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "application/pdf")}
        resp = requests.post(url, headers=HEADERS, files=files, timeout=120)
    if resp.status_code == 200:
        return True, resp.json()
    return False, resp.text

def watch_and_ingest():
    for root, _, files in os.walk(CLEAN_DIR):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                full = os.path.join(root, fn)
                ok, res = upload_doc(full)
                print("OK" if ok else "FAIL", fn, res)
                time.sleep(0.5)

if __name__ == "__main__":
    watch_and_ingest()
```
Refer to AnythingLLM’s API docs for exact endpoints and auth flows. [7][3]

***

## [2]. Handling large-file failures

Very large PDFs may fail in automated embedding. Those are routed to a designated local folder for manual upload through the AnythingLLM GUI. This hybrid flow ensures no document is blocked by API constraints. [3]

- Example Python router:
```python
import os, shutil

FAILED_DIR = r"D:\RAG\failed_manual"

def route_failure(path):
    os.makedirs(FAILED_DIR, exist_ok=True)
    dst = os.path.join(FAILED_DIR, os.path.basename(path))
    shutil.move(path, dst)
    print("Routed to manual:", dst)
```

- Manual GUI upload:
  - Open AnythingLLM, select the workspace, and use the Documents UI to add the large PDF directly. [3]
***

## [3] Complete REST Ingestion code

Pasted below is the complete the code block for python watcher.py:-

```
#!/usr/bin/env python3

import os
import sys
import time
import json
import math
import shutil
import requests
from pathlib import Path

CONFIG_PATH = "watcher_config.json"
LOG_PATH = "embedded_log.json"

# -------------------------------------------------------------
# Load configuration
# -------------------------------------------------------------
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

WATCH_FOLDER = Path(cfg["watch_folder"])
API_URL = cfg["anythingllm_url"].rstrip("/")
API_KEY = cfg["api_key"]
WORKSPACE = cfg["workspace"]  # workspace slug
MAX_BATCH_MB = cfg.get("max_batch_mb", 20)
CHECK_INTERVAL = cfg.get("check_interval_sec", 60)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# -------------------------------------------------------------
# Docker / RAM settings
# -------------------------------------------------------------
DOCKER_API = "http://localhost:2375"
CONTAINER = "anythingllm"

RAM_PAUSE = 80.0    # pause when RAM >= 80%
RAM_RESUME = 65.0   # resume when RAM <= 65%

RESTART_WAIT = 30   # seconds to wait after each restart
RETRY_DELAY = 120   # seconds between upload retries
UPLOAD_TIMEOUT = 60 # per attempt HTTP timeout
MAX_RETRIES = 3

# Error folder
ERROR_FOLDER = Path(r"C:\Users\biswa\Downloads\error jobs")
ERROR_FOLDER.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# Progress bar helpers
# -------------------------------------------------------------
def update_progress_bar(done, total, batch_idx, batch_total, current_pdf, ram=None, waiting=False):
    """
    Render a simple single-line progress bar in the terminal.
    """
    if total <= 0:
        total = 1
    pct = (done / total) * 100.0

    bar_len = 20
    filled_len = int(bar_len * pct / 100.0)
    bar = "█" * filled_len + "-" * (bar_len - filled_len)

    status = "WAITING" if waiting else "RUNNING"
    ram_text = f" | RAM: {ram}%" if ram is not None else ""

    msg = (
        f"\r[{bar}] {pct:5.1f}% ({done}/{total} PDFs)"
        f" | Batch {batch_idx}/{batch_total}"
        f" | {status}"
        f" | Current: {current_pdf}{ram_text}   "
    )

    sys.stdout.write(msg)
    sys.stdout.flush()


def clear_progress_bar():
    sys.stdout.write("\r" + " " * 140 + "\r")
    sys.stdout.flush()


# -------------------------------------------------------------
# Logging Helpers
# -------------------------------------------------------------
def load_log():
    if Path(LOG_PATH).exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_log(log):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


# -------------------------------------------------------------
# Error Handling
# -------------------------------------------------------------
def move_to_error(pdf, reason="Unknown error"):
    try:
        target = ERROR_FOLDER / pdf.name
        shutil.move(str(pdf), str(target))
        print(f"\n🚫 ERROR → {pdf.name} moved to error folder ({reason})")
    except Exception as e:
        print(f"\n⚠️ Failed to move {pdf.name} to error folder: {e}")


# -------------------------------------------------------------
# Docker Container Helpers
# -------------------------------------------------------------
def is_container_running():
    try:
        r = requests.get(f"{DOCKER_API}/containers/{CONTAINER}/json", timeout=5)
        if r.status_code != 200:
            print(f"⚠️ Container inspect HTTP {r.status_code}")
            return False
        state = r.json().get("State", {})
        return bool(state.get("Running", False))
    except Exception as e:
        print(f"⚠️ Container check error: {e}")
        return False


def restart_container_twice():
    print("\n⛔ Container is NOT running. Performing double restart for safety...")
    for i in range(2):
        try:
            print(f"♻️ Restart {i+1}/2...")
            r = requests.post(
                f"{DOCKER_API}/containers/{CONTAINER}/restart",
                timeout=10
            )
            if r.status_code not in (200, 204):
                print(f"⚠️ Restart {i+1} HTTP {r.status_code}")
        except Exception as e:
            print(f"❌ Restart {i+1} failed: {e}")
        print(f"⏳ Waiting {RESTART_WAIT} seconds after restart {i+1}...")
        time.sleep(RESTART_WAIT)
    print("✅ Double restart complete.")


# -------------------------------------------------------------
# RAM Monitoring
# -------------------------------------------------------------
def get_ram_usage():
    try:
        r = requests.get(
            f"{DOCKER_API}/containers/{CONTAINER}/stats?stream=false",
            timeout=5
        )
        if r.status_code != 200:
            print(f"⚠️ RAM stats HTTP {r.status_code}")
            return None
        js = r.json()
        used = js["memory_stats"]["usage"]
        limit = js["memory_stats"]["limit"]
        if not limit:
            return None
        pct = used / limit * 100.0
        return round(pct, 2)
    except Exception as e:
        print(f"⚠️ RAM check error: {e}")
        return None


def wait_for_safe_ram(global_done=0, global_total=1, batch_idx=0, batch_total=1, label="RAM WAIT"):
    """
    Enforce:
      - pause uploads when RAM >= RAM_PAUSE
      - resume when RAM <= RAM_RESUME
    Also ensures the container is running (double restart if needed).
    """
    # Ensure container is alive first
    if not is_container_running():
        restart_container_twice()

    while True:
        ram = get_ram_usage()
        if ram is None:
            print("⚠️ Could not read RAM usage, retrying in 5s...")
            time.sleep(5)
            continue

        update_progress_bar(
            global_done,
            global_total,
            batch_idx,
            batch_total,
            current_pdf=label,
            ram=ram,
            waiting=(ram >= RAM_PAUSE),
        )

        if ram < RAM_PAUSE:
            # clear line once we are safe again
            clear_progress_bar()
            return

        print(f"\n⛔ RAM too high ({ram}% ≥ {RAM_PAUSE}%). Waiting for ≤ {RAM_RESUME}%...")

        # Inner loop: stay here until RAM safe
        while True:
            if not is_container_running():
                print("⛔ Container stopped while waiting on RAM. Restarting twice...")
                restart_container_twice()

            ram2 = get_ram_usage()
            if ram2 is None:
                time.sleep(5)
                continue

            update_progress_bar(
                global_done,
                global_total,
                batch_idx,
                batch_total,
                current_pdf=label,
                ram=ram2,
                waiting=True,
            )

            if ram2 <= RAM_RESUME:
                print(f"\n✅ RAM safe again ({ram2}% ≤ {RAM_RESUME}%). Resuming uploads.")
                clear_progress_bar()
                return

            time.sleep(5)


# -------------------------------------------------------------
# Upload-only logic (Option A, no embedding verification)
# -------------------------------------------------------------
def upload_pdf(pdf_path, global_done, global_total, batch_idx, batch_total):

    def attempt_upload():
        # Ensure RAM + container OK before starting
        wait_for_safe_ram(
            global_done=global_done,
            global_total=global_total,
            batch_idx=batch_idx,
            batch_total=batch_total,
            label=f"Uploading {pdf_path.name}",
        )

        ram = get_ram_usage()
        update_progress_bar(
            global_done,
            global_total,
            batch_idx,
            batch_total,
            current_pdf=f"Uploading {pdf_path.name}",
            ram=ram,
            waiting=False,
        )

        files = {
            "file": (pdf_path.name, open(pdf_path, "rb"), "application/pdf"),
        }
        data = {
            "addToWorkspaces": WORKSPACE,
            "metadata": json.dumps({
                "title": pdf_path.stem,
                "docAuthor": "AutoUploader",
                "description": "Uploaded automatically via watcher script",
                "docSource": str(pdf_path),
            }),
        }
        url = f"{API_URL}/api/v1/document/upload"

        try:
            response = requests.post(
                url,
                files=files,
                data=data,
                headers=HEADERS,
                timeout=UPLOAD_TIMEOUT,
            )
            return response
        finally:
            try:
                files["file"][1].close()
            except Exception:
                pass

    print(f"\n⬆️ Uploading: {pdf_path.name}")

    # Retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        # Container health check each attempt
        if not is_container_running():
            print("⛔ Container is down before upload attempt. Restarting twice...")
            restart_container_twice()

        try:
            response = attempt_upload()
            break
        except Exception as e:
            clear_progress_bar()
            print(f"⚠️ Upload attempt {attempt}/{MAX_RETRIES} failed for {pdf_path.name}: {e}")
            if attempt < MAX_RETRIES:
                print(f"⏳ Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
    else:
        move_to_error(pdf_path, "Upload failed after retries")
        return False

    # Validate response JSON
    try:
        res_json = response.json()
    except Exception:
        clear_progress_bar()
        print(f"❌ Invalid JSON response: {response.text[:500]}")
        move_to_error(pdf_path, "Invalid JSON response")
        return False

    if response.status_code != 200 or not res_json.get("success"):
        clear_progress_bar()
        print(
            f"❌ Upload API failure for {pdf_path.name} "
            f"(HTTP {response.status_code}, success={res_json.get('success')})"
        )
        move_to_error(pdf_path, f"API error {response.status_code}")
        return False

    clear_progress_bar()
    print(f"✅ Uploaded OK (embedding will happen internally): {pdf_path.name}")
    return True


# -------------------------------------------------------------
# Batching logic
# -------------------------------------------------------------
def batch_pdfs(pdfs):
    """
    Yield lists of PDFs whose total size is <= MAX_BATCH_MB.
    """
    batch, total_mb = [], 0.0
    for pdf in pdfs:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        if total_mb + size_mb > MAX_BATCH_MB and batch:
            yield batch
            batch, total_mb = [], 0.0
        batch.append(pdf)
        total_mb += size_mb
    if batch:
        yield batch


# -------------------------------------------------------------
# Pending PDFs based on log
# -------------------------------------------------------------
def get_pending_pdfs():
    log = load_log()
    all_pdfs = sorted(WATCH_FOLDER.glob("*.pdf"))
    return [p for p in all_pdfs if str(p) not in log]


# -------------------------------------------------------------
# File stability check
# -------------------------------------------------------------
def is_file_stable(path: Path, wait: int = 2) -> bool:
    """
    Returns True if the file size is unchanged over 'wait' seconds.
    Helps avoid partially copied files.
    """
    try:
        size1 = path.stat().st_size
        time.sleep(wait)
        size2 = path.stat().st_size
        return size1 == size2
    except FileNotFoundError:
        return False


# -------------------------------------------------------------
# Main Watcher Loop
# -------------------------------------------------------------
def main():
    print(f"👀 Watching folder: {WATCH_FOLDER}")
    log = load_log()

    while True:
        pending = get_pending_pdfs()

        if pending:
            total_pdfs = len(pending)
            print(f"\n🔍 Found {total_pdfs} new PDF(s).")

            batches = list(batch_pdfs(pending))
            total_batches = len(batches)
            print(f"📦 Total batches to process: {total_batches}")

            processed_count = 0

            for batch_idx, batch in enumerate(batches, start=1):
                print(
                    f"\n📦 Processing batch {batch_idx}/{total_batches} "
                    f"({len(batch)} PDF(s))..."
                )

                for pdf in batch:
                    processed_count_display = processed_count  # before this PDF

                    # Show pre-upload progress bar
                    update_progress_bar(
                        done=processed_count_display,
                        total=total_pdfs,
                        batch_idx=batch_idx,
                        batch_total=total_batches,
                        current_pdf=pdf.name,
                        ram=get_ram_usage(),
                        waiting=False,
                    )

                    # Ensure file is stable
                    if not is_file_stable(pdf):
                        clear_progress_bar()
                        print(f"⚠️ File not stable yet, retrying in 5s: {pdf.name}")
                        time.sleep(5)
                        if not is_file_stable(pdf):
                            print(f"❌ File still unstable, skipping for now: {pdf.name}")
                            continue

                    success = upload_pdf(
                        pdf,
                        global_done=processed_count_display,
                        global_total=total_pdfs,
                        batch_idx=batch_idx,
                        batch_total=total_batches,
                    )

                    processed_count += 1

                    # Update log
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    log[str(pdf)] = ts if success else f"FAILED - {ts}"
                    save_log(log)

                    clear_progress_bar()

            print("\n✅ All pending PDFs in this cycle processed.")
        else:
            print("⏳ No new PDFs. Sleeping...")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

```
***

