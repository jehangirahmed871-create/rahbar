# Corpus Specification and Manifest Analysis

**Source:** FAST-NUCES Public Website (`nu.edu.pk`)  
**Curator:** Jehangir Ahmed (Person A — Data and Quality)  
**Total Documents:** 9 fetched documents  
**Total Pages:** 279+ pages (well exceeding the minimum 40-page requirement)  
**Manifest:** Recorded in [`corpus/MANIFEST.md`](../corpus/MANIFEST.md) with verified SHA-256 checksums  

---

## 1. Overview and Required Properties Mapping

The corpus is selected from the official public portal of FAST-NUCES (`https://nu.edu.pk`) to provide realistic regulatory and operational university documents. The collection intentionally introduces real-world documentation challenges to make every target RAG failure mode reachable and verifiable.

| # | Required Property | Document(s) | Version / Metadata | Evidence & Failure Mode Analysis |
|---|---|---|---|---|
| **1** | **At least 6 documents and 40 pages in total** | All 9 documents in `corpus/sources.yaml` | Multi-document corpus | 9 documents totaling 279+ pages. Scales the corpus so vector indexing and retrieval top-k selectivity matter. |
| **2** | **Two versions of one policy** | `undergraduate-academic-rules-aug-2023.pdf` vs `academic-rules-handbook-jun-2026.pdf` | `2023-08` (`superseded: true`) vs `2026-06` (`superseded: false`) | **Failure Mode E (Outdated Source):** Both describe prerequisite registration and attendance rules. Without metadata filtering on `superseded: false`, the retriever returns outdated 2023 clauses instead of current 2026 rules. |
| **3** | **Scanned, image-only PDF with no text layer** | `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` | `2022-05` (`superseded: false`) | **Failure Mode A (Bad Extraction):** Running standard PDF text extraction (`pypdf`/`pdfminer`) returns 0 characters across all 8 pages (`[0, 0, 0, 0, 0, 0, 0, 0]`). The document requires OCR via Tesseract/pdf2image to be indexable. |
| **4** | **Document with a table carrying real information** | `fee-structure-page.html`, `grading-page.html`, `annual-report-2023-24.pdf` | HTML / PDF tabular data | **Failure Mode A (Mangled Tables):** HTML tables and annual report faculty/graduation tables. Naive spatial text extraction concatenates columns across rows; structure-aware extraction preserves column-row relationships. |
| **5** | **Long definitions split by fixed chunkers** | `student-handbook-jun-2026.pdf`, `academic-rules-handbook-jun-2026.pdf` | `2026-06` | **Failure Mode B (Poor Chunks):** Contains extensive multi-paragraph definitions of Academic Dishonesty, Generative AI acceptable usage policy, and probation regulations. Fixed 100-token chunks sever condition from consequence; structure-aware chunking preserves section headings. |
| **6** | **Deliberately absent topic** | Hostel Room Allotment and Transport Fee Structure | Topic deliberately omitted | **Abstention (Honest Refusal):** Prospectus PDF `Transport and Hostel Facilities.pdf` was intentionally excluded from `sources.yaml`. When asked about hostel application rules or fees, the model must return `answered_from_context: false` rather than hallucinating generic university hostel rules. |
| **7** | **Exact-term content** | `academic-rules-handbook-jun-2026.pdf`, `prospectus-2026-academic-rules-and-regulations.pdf`, `fee-structure-page.html` | Clause numbers, CGPA thresholds, fees | **Embeddings blur exact terms:** Clause numbers (e.g. Clause 1.21 attendance in Handbook vs Clause 1 in Prospectus), CGPA warning threshold (CGPA < 2.00 in Clause 5.4), credit hour fee rates (e.g., PKR tuition rates). Dense vector embeddings map "Clause 1.21" close to "Clause 1.22", demonstrating the necessity of strict relevance filtering. |

---

## 2. Detailed Document Inventory

### 2.1 Academic Rules Handbook (Jun 2026) — Current Policy
- **File:** `academic-rules-handbook-jun-2026.pdf`
- **Size / Pages:** 987,731 bytes / 58 pages
- **Metadata:** `doc_version: "2026-06"`, `superseded: false`
- **Role:** The primary ground truth for undergraduate student policies. Covers course registration, withdrawal deadlines, prerequisite rules, grade replacements, warning thresholds (Clause 5.4: CGPA < 2.00), and 80% mandatory attendance threshold (Clause 1.21).

### 2.2 Undergraduate Academic Rules (Aug 2023) — Superseded Policy
- **File:** `undergraduate-academic-rules-aug-2023.pdf`
- **Size / Pages:** 14,888,226 bytes / 26 pages
- **Metadata:** `doc_version: "2023-08"`, `superseded: true`
- **Role:** Historical policy retained on the university site. Contains obsolete repeat course grade capping clauses and older warning criteria. Used specifically to test the retriever's metadata filtering (`superseded: false`).

### 2.3 Prospectus 2026: Academic Rules and Regulations
- **File:** `prospectus-2026-academic-rules-and-regulations.pdf`
- **Size / Pages:** 7,066,767 bytes / 5 pages
- **Metadata:** `doc_version: "2026"`, `superseded: false`
- **Role:** Condensed summary of academic rules. Tests cross-document consolidation where identical rules are referenced under distinct clause numbering (e.g., Clause 1 vs Clause 1.21 in Handbook).

### 2.4 Faculty Entrepreneurship and Technical Start-ups Policy — Scanned Document
- **File:** `faculty-entrepreneurship-and-technical-start-ups-policy.pdf`
- **Size / Pages:** 3,872,809 bytes / 8 pages
- **Metadata:** `doc_version: "2022-05"`, `superseded: false`
- **Proof of Image-Only Scan:** Text extraction using `pypdf` yields exactly `0` bytes for all pages (`[0, 0, 0, 0, 0, 0, 0, 0]`). Inspection via `pdffonts` confirms an empty font dictionary, indicating full-page rasterized bitmaps. Indexing this document without OCR results in 0 chunks and complete retrieval blindness (evaluated in Failure Stage A).

### 2.5 Student Handbook (Jun 2026)
- **File:** `student-handbook-jun-2026.pdf`
- **Size / Pages:** 1,980,275 bytes / 64 pages
- **Metadata:** `doc_version: "2026-06"`, `superseded: false`
- **Role:** Code of conduct, student discipline, library rules, and the multi-section Generative AI acceptable usage guidelines. Contains deeply nested sub-sections where arbitrary fixed-token splitting separates definitions from their operational constraints.

### 2.6 Annual Reports (2019-20 and 2023-24)
- **Files:** `annual-report-2019-20.pdf` (33 pages) and `annual-report-2023-24.pdf` (85 pages)
- **Metadata:** `doc_version: "2019-20"` and `"2023-24"`, `superseded: false`
- **Role:** Quantitative institutional data in complex multi-column tables detailing faculty counts, publications by campus, research grants, and graduate statistics across multiple academic years.

### 2.7 HTML Tabular Documents: Fee Structure and Grading
- **Files:** `fee-structure-page.html` (91,429 bytes) and `grading-page.html` (95,673 bytes)
- **Metadata:** `doc_version: "2026-27"` and `"2026"`, `superseded: false`
- **Role:** Extracted HTML tables representing semester fees, credit hour tuition rates, admission charges, refund percentage schedules, and the letter-grade to grade-point (4.00 scale) conversion table.

---

## 3. Deliberately Absent Topic and Abstention Strategy

To evaluate the system's ability to decline questions outside its ground truth, **Hostel Accommodation and University Transport Regulations** have been intentionally omitted:
- **Exclusion:** The official prospectus booklet `Transport and Hostel Facilities.pdf` was excluded from `corpus/sources.yaml`.
- **Trap:** The `Student Handbook` mentions that university hostels and bus routes exist, but does not specify fee schedules, allocation quotas, curfews, or application procedures.
- **Evaluation:** Golden questions `q13` and `q14` ask: *"What is the procedure and semester fee for securing a room in the university boys' hostel?"*
- **Required Behavior:** The retriever top-k returns low relevance scores below the relevance floor (`0.50`), and the API generates:
  ```json
  {
    "answered_from_context": false,
    "confidence": "low",
    "answer": "The provided corpus documents do not contain information regarding hostel accommodation procedures or fees.",
    "citations": []
  }
  ```
  Answering this question from general pre-trained parametric memory is treated as an automatic failure (-10 marks).

---

## 4. Integrity and Reproducibility

1. **Manifest Verification:** All documents are registered in `corpus/MANIFEST.md` with SHA-256 hashes.
2. **Polite Download Protocol:** `corpus/fetch.py` executes sequential requests separated by 2.0-second delay intervals with a dedicated student User-Agent string to prevent denial-of-service on the host servers.
3. **Storage Discipline:** The raw downloaded PDFs in `corpus/raw/` are excluded from version control via `.gitignore`. The repository only commits `corpus/sources.yaml`, `corpus/MANIFEST.md`, and the automated fetch script.
