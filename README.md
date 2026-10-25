# 🚗 InsuranceAI

### Multimodal AI-Powered Vehicle Damage Assessment & Insurance Report Generator

InsuranceAI is an end-to-end AI system that:

* Detects vehicle damage using YOLOv8 segmentation
* Generates structured insurance damage reports using LLMs
* Creates professional PDF reports
* Visualizes damage masks with labeled overlays
* Designed for integration with Agentic AI systems (LangGraph-ready)

---

## 🎥 Demo

https://github.com/Sanjeevspuranik/InsuranceAI/blob/main/demo/video/InsuranceAI_demo.mp4


## 🏗️ System Architecture

![InsuranceAI System Design](demo/system_design.svg)


## 📄 Sample Generated Report

[Download Sample Insurance Report](demo/document/damage_report.pdf)

---

## 🔥 Features

* ✅ YOLOv8 Segmentation-based Damage Detection
* ✅ Bounding Box + Mask Predictions
* ✅ GPT-based Insurance Damage Report Generation
* ✅ Formatted PDF reports with section styling, page numbers, and annotated evidence
* ✅ Class-wise Segmentation Visualization
* ✅ Modular Architecture for SaaS / API Deployment

---

# 🧠 Damage Classes

| Class ID | Damage Type   |
| -------- | ------------- |
| 0        | Dent          |
| 1        | Scratch       |
| 2        | Crack         |
| 3        | Glass Shatter |
| 4        | Lamp Broken   |
| 5        | Tire Flat     |

---

# 📊 Model Details

### Task

`segment` (Instance Segmentation)

### Framework

* Ultralytics YOLOv8
* Version: 8.4.14
* Python: 3.12.3
* Torch: 2.10.0 (CPU)
* Hardware: AMD Ryzen 9 6900HS (CPU training/inference)

---

## 📈 Validation Metrics

### Overall Performance

| Metric    | Bounding Box | Mask  |
| --------- | ------------ | ----- |
| Precision | 0.734        | 0.745 |
| Recall    | 0.684        | 0.691 |
| mAP@50    | 0.723        | 0.720 |
| mAP@50-95 | 0.556        | 0.532 |

**Fitness Score:** `1.0877`

---

## 📊 Class-wise Performance

| Class         | Images | Instances | Box mAP50 | Mask mAP50 |
| ------------- | ------ | --------- | --------- | ---------- |
| Dent          | 157    | 236       | 0.629     | 0.627      |
| Scratch       | 183    | 307       | 0.592     | 0.604      |
| Crack         | 48     | 70        | 0.409     | 0.396      |
| Glass Shatter | 71     | 71        | 0.982     | 0.984      |
| Lamp Broken   | 65     | 69        | 0.815     | 0.802      |
| Tire Flat     | 31     | 32        | 0.910     | 0.910      |

---

## ⚡ Inference Speed

| Stage       | Time per Image |
| ----------- | -------------- |
| Preprocess  | 0.9 ms         |
| Inference   | 54.9 ms        |
| Postprocess | 5.1 ms         |

~60ms per image (CPU)

---

# 🏗️ Model Fine-Tuning Details

### Dataset

* 374 validation images
* 785 total instances
* Balanced across 6 vehicle damage classes
* YOLO format labels (segmentation masks)

### Dataset Distribution

```
dent: 236
scratch: 307
crack: 70
glass shatter: 71
lamp broken: 69
tire flat: 32
```

---

### Training Configuration (YOLOv8)

Typical configuration used:

```
Task: segment
Image size: 640
Batch size: (hardware dependent)
Optimizer: SGD / AdamW (default Ultralytics)
Loss:
  - Box loss
  - Mask loss
  - Classification loss
Augmentations:
  - Mosaic
  - HSV augmentation
  - Horizontal flip
  - Scaling
```

Model checkpoint:

```
model/CarDD_FN_best.pt
```

Validation results saved at:

```
runs/segment/val
```

---

# 🏛️ Architecture Overview

```
User Image
    ↓
YOLOv8 Segmentation
    ↓
Structured JSON Extraction
    ↓
LLM Damage Analysis
    ↓
PDF Report Generator
    ↓
Final Insurance Report
```

---

# 📂 Project Structure

```
InsuranceAI/
│
├── model/
│   └── CarDD_FN_best.pt
│
├── dataset/
│   ├── images/
│   └── labels/
│
├── prompts.py
├── your_module.py
├── app.py
└── README.md
```

---

# ⚙️ Installation

Use Python 3.12 and Node.js 20.9 or newer. Install the Python dependencies and
create the root environment file. If `.env`
already exists, merge the needed settings from [.env.example](./.env.example)
instead of overwriting it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

On macOS or Linux, activate `.venv` and use `cp .env.example .env`.

Set `OPENAI_API_KEY` in `.env`. For local API authentication, keep
`API_AUTH_MODE=development` and set a private `API_DEV_TOKEN`.
Initialize the local SQLite schema once:

```powershell
alembic upgrade head
```

If you are upgrading a local database created by an earlier version that
automatically created the original `claims` table, back it up first, then run
`alembic stamp 0001_create_claims` followed by `alembic upgrade head`. Only stamp
databases that already contain that original schema; fresh databases should use
`alembic upgrade head` directly.

## Run the API, worker, and web app

Open three terminals in the repository root.

**Terminal 1 — API**

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — analysis worker**

```powershell
.\.venv\Scripts\Activate.ps1
python -m api.worker
```

**Terminal 3 — Next.js**

```powershell
if (!(Test-Path frontend\.env.local)) { Copy-Item frontend\.env.example frontend\.env.local }
```

Set `API_DEV_TOKEN` in `frontend\.env.local` to the exact same value as the
root `.env`. If browsing from another device over the LAN, set
`NEXT_ALLOWED_DEV_ORIGINS` to the dev server host or IP (for example,
`192.168.31.69`) to permit Next.js HMR requests. Generate a local NextAuth secret and set `NEXTAUTH_SECRET` in
`frontend\.env.local`:

```powershell
node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"
```

Then run:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open `http://localhost:3000`, choose **Sign in (local development)**, then upload
a vehicle image. The API stores queued claims and local artifacts under the
ignored `data/` directory. The API and worker use the same `DATABASE_URL`.
`main.py` remains the CLI entry point; `interface.py` is retained as the legacy
Gradio UI and is not required to run the FastAPI/Next.js application.
Backend logs include a request ID for each HTTP request and a claim ID across
worker processing. Set `LOG_LEVEL=DEBUG` in `.env` for request-start and
pipeline-stage detail. Request logs omit bodies and auth headers; application
events do not log uploaded image or report contents. Logs are written to the
repository-root `server.log` and the console; the file rotates at 10 MB and
retains five backups.

### OIDC deployment

For a deployed environment, set `APP_ENV=production` and
`API_AUTH_MODE=oidc` in the API environment. Configure `OIDC_ISSUER_URL`,
`OIDC_JWKS_URL`, and `OIDC_AUDIENCE` for the API, and `OIDC_ISSUER_URL`,
`OIDC_CLIENT_ID`, and `OIDC_CLIENT_SECRET` for the Next.js app. Configure the
identity provider to issue an API access token with the configured audience;
add tenant and adjuster/admin role claims if those scopes are used.
Development-token authentication is explicitly disabled when `APP_ENV=production`.
Apply schema migrations against the configured production database before
starting API or worker processes:

```powershell
alembic upgrade head
```

### Replace local providers

The default local database is SQLite. Change `DATABASE_URL` to a SQLAlchemy
PostgreSQL URL (for example `postgresql://user:password@host:5432/insuranceai`)
to use PostgreSQL; install the PostgreSQL driver in that environment and migrate
existing data separately. Claims and queued jobs share the configured database.

Change `STORAGE_URL` from `file:./data/storage` to `s3://bucket/prefix` for
S3-compatible object storage. Supply AWS-compatible credentials through the
environment, and set `S3_ENDPOINT_URL` when using a non-AWS endpoint. The
frontend API origin is configured by `NEXT_PUBLIC_API_BASE_URL`.

## Run tests

```powershell
python -m unittest discover -s tests
Set-Location frontend
npm run typecheck
npm run build
```

---

# 🚀 Usage

### 1️⃣ Run Damage Detection

```python
from your_module import predict_image

result = predict_image(img_path="car.jpg")
print(result)
```

---

### 2️⃣ Generate Damage Report

```python
from your_module import generate_damage_report

report = generate_damage_report(result)
print(report)
```

---

### 3️⃣ Generate PDF

```python
from your_module import create_damage_pdf

create_damage_pdf(report)
```

---

### 4️⃣ Visualize Segmentation

```python
from your_module import visualize_segments
from ultralytics import YOLO

model = YOLO("model/CarDD_FN_best.pt")
results = model.predict("car.jpg")

visualize_segments(results)
```

---

# 🧠 LLM Integration

* Uses GPT-6-Luna through `langchain-openai` (configurable with `OPENAI_MODEL`)
* Sends the uploaded JPEG, PNG, or WEBP image as a base64 data URL in a multimodal
  `HumanMessage`, alongside YOLO detections
* Produces a structured damage report and a best-effort, uncertainty-qualified
  visual estimate of the vehicle make and model
* Displays provider-reported reasoning-token counts when available, without
  exposing hidden chain-of-thought text
* Does not generate monetary estimates; pricing research is not configured
* Displays a concise assessment rationale from the final response, not hidden
  chain-of-thought or raw reasoning tokens

Set `OPENAI_API_KEY` in the environment or `.env` file before starting the app.
For an OpenAI-compatible provider, configure `OPENAI_BASE_URL` as required by that
provider. Set `OPENAI_MODEL` to override the default `gpt-6-luna` model identifier.

---

# 🔮 Future Roadmap

* [ ] Cost estimation engine per damage class
* [ ] Human-in-the-loop claim approval
* [x] LangGraph agentic workflow
* [ ] Multi-image claim analysis
* [x] Authenticated FastAPI API
* [x] Next.js web application
* [ ] Mobile upload integration

---

# 🏆 Why InsuranceAI?

* Accurate segmentation-based detection
* High performance on critical classes (Glass, Tire, Lamp)
* CPU-compatible
* Modular production-ready architecture
* Designed for real-world insurance automation

---

# 👨‍💻 Author

**Sanjeev S Puranik**
AI Engineer | Multimodal Systems | Agentic AI
