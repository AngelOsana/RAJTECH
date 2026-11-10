"""Kaagapay backend (Render): retrieval from Supabase pgvector + gpt-oss-120b on Cerebras + chat history in Supabase."""
import os, re, time, uuid, threading
import httpx, numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ["SUPABASE_KEY"]                       # service_role key, kept only here (Render environment)
LLM_MODEL = os.environ.get("LLM_MODEL", 'gpt-oss-120b')
TOP_K = int(os.environ.get("TOP_K", 3))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
_thr = os.environ.get("FALLBACK_THRESHOLD", None)
FALLBACK_THRESHOLD = float(_thr) if _thr not in (None, "", "None") else None
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # same model as the notebook (ONNX version, no PyTorch)
SYSTEM_PROMPT = 'You are Kaagapay, a helpful assistant that answers questions about social protection for poor families in the Philippines.\nRules:\n1. Answer ONLY from the CONTEXT below. Do not use outside knowledge and do not invent facts, numbers, dates, names or eligibility decisions.\n2. Reply in the same language as the question (English, Tagalog or Taglish).\n3. Mention the source document (and page) that supports your answer.\n4. If the CONTEXT does not contain the answer, say clearly that the documents do not contain it and refer the user to their local DSWD office. Do not guess.'
FALLBACK_MESSAGE = 'Sorry, I could not find the answer in my documents. / Pasensya na, hindi ko makita ang sagot sa mga dokumentong mayroon ako. Please contact your local DSWD office or visit the official DSWD website.'

HEADERS = {"apikey": SUPABASE_KEY, "Authorization": "Bearer " + SUPABASE_KEY, "Content-Type": "application/json"}
http = httpx.Client(timeout=30)
llm = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=os.environ["CEREBRAS_API_KEY"])

app = FastAPI(title="Kaagapay API")
app.add_middleware(CORSMiddleware, allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
                   allow_methods=["*"], allow_headers=["*"])

_embedder = None
_lock = threading.Lock()
def embed(text):
    global _embedder
    with _lock:
        if _embedder is None:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(EMBEDDING_MODEL)
    v = np.array(list(_embedder.embed([text]))[0], dtype=float)
    n = np.linalg.norm(v)
    return (v / (n if n else 1.0)).round(6).tolist()

@app.on_event("startup")
def warm_up():
    threading.Thread(target=lambda: embed("warm up"), daemon=True).start()

def sb(method, path, **kw):
    r = http.request(method, SUPABASE_URL + "/rest/v1/" + path, headers=kw.pop("headers", HEADERS), **kw)
    if r.status_code >= 400:
        raise HTTPException(502, "Database error: " + r.text[:200])
    return r

CLIENT_RE = re.compile(r"^[A-Za-z0-9-]{16,64}$")
def check_client(cid):
    if not CLIENT_RE.match(cid or ""):
        raise HTTPException(400, "Invalid client id")
    return cid
def check_conv(conv_id, client_id):
    try:
        uuid.UUID(conv_id)
    except Exception:
        raise HTTPException(400, "Invalid conversation id")
    rows = sb("GET", "conversations", params={"id": "eq." + conv_id, "client_id": "eq." + client_id, "select": "id"}).json()
    if not rows:
        raise HTTPException(404, "Conversation not found")

def retrieve(question):
    r = sb("POST", "rpc/match_chunks", json={"query_embedding": embed(question), "match_count": TOP_K})
    return [{"source": h["source"], "page": h["page"], "text": h["content"], "similarity": round(h["similarity"], 3)} for h in r.json()]

def generate(messages):
    for attempt in range(4):
        try:
            r = llm.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.0,
                                            max_completion_tokens=MAX_TOKENS, reasoning_effort="low")
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e).lower()
            if attempt < 3 and any(k in msg for k in ("429", "rate", "queue", "overloaded", "503")):
                time.sleep(13 * (attempt + 1))
                continue
            raise

class ChatIn(BaseModel):
    client_id: str
    message: str
    conversation_id: str | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL, "top_k": TOP_K}

@app.get("/conversations")
def list_conversations(client_id: str = Query(...)):
    check_client(client_id)
    return sb("GET", "conversations", params={"client_id": "eq." + client_id, "select": "id,title,created_at",
                                              "order": "created_at.desc", "limit": "50"}).json()

@app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str, client_id: str = Query(...)):
    check_client(client_id); check_conv(conv_id, client_id)
    return sb("GET", "messages", params={"conversation_id": "eq." + conv_id, "select": "role,content,sources",
                                         "order": "id.asc"}).json()

@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, client_id: str = Query(...)):
    check_client(client_id); check_conv(conv_id, client_id)
    sb("DELETE", "conversations", params={"id": "eq." + conv_id, "client_id": "eq." + client_id})
    return {"deleted": conv_id}

@app.post("/chat")
def chat(body: ChatIn):
    check_client(body.client_id)
    question = body.message.strip()[:500]
    if not question:
        raise HTTPException(400, "Empty message")
    conv_id = body.conversation_id
    if conv_id:
        check_conv(conv_id, body.client_id)
    else:
        created = sb("POST", "conversations", json={"client_id": body.client_id, "title": question[:60]},
                     headers={**HEADERS, "Prefer": "return=representation"}).json()
        conv_id = created[0]["id"]
    sb("POST", "messages", json={"conversation_id": conv_id, "role": "user", "content": question},
       headers={**HEADERS, "Prefer": "return=minimal"})

    hits = retrieve(question)
    sources = [{"source": h["source"], "page": h["page"], "similarity": h["similarity"]} for h in hits]
    if FALLBACK_THRESHOLD is not None and hits and hits[0]["similarity"] < FALLBACK_THRESHOLD:
        answer = FALLBACK_MESSAGE                                   # retrieval gate: decline without calling the LLM
    else:
        context = "\n\n".join("[Source: %s, p.%s]\n%s" % (h["source"], h["page"], h["text"]) for h in hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "CONTEXT:\n%s\n\nQUESTION: %s\n\nANSWER:" % (context, question)}]
        try:
            answer = generate(messages)
        except Exception:
            raise HTTPException(503, "The language model is busy or its limit was reached. Please try again in a minute.")
    sb("POST", "messages", json={"conversation_id": conv_id, "role": "assistant", "content": answer, "sources": sources},
       headers={**HEADERS, "Prefer": "return=minimal"})
    return {"conversation_id": conv_id, "answer": answer, "sources": sources}
