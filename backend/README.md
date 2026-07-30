# Agentverse Resume Tailor Backend

FastAPI backend for a 7-node LangGraph pipeline that tailors a resume and cover letter to a target company using Grok via the xAI API.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `XAI_API_KEY` in `.env`, then run:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Endpoint

`POST /api/tailor`

```json
{
  "resume_text": "Raw resume text...",
  "job_input": "Job post text or URL",
  "job_input_type": "text",
  "blog_urls": [],
  "github_org": null
}
```

## Curl Example

```bash
curl -X POST http://127.0.0.1:8000/api/tailor ^
  -H "Content-Type: application/json" ^
  -d "{\"resume_text\":\"Jane Doe\nPython developer...\",\"job_input\":\"We use FastAPI, LangGraph, and cloud APIs.\",\"job_input_type\":\"text\",\"blog_urls\":[],\"github_org\":null}"
```

## Render Deploy

1. Create a new Render Web Service.
2. Connect this repository.
3. Use the included `Procfile`.
4. Add `XAI_API_KEY` as an environment variable.
5. Deploy.

Render will run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```




