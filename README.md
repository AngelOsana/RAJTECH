# Kaagapay chatbot (ITC 508, Group RAJTECH)

RAG chatbot on social protection documents (RA 11291, RA 11310 IRR, SP Operational Framework).

- `backend/`  FastAPI service for Render: retrieval (Supabase pgvector) + gpt-oss-120b (Cerebras) + chat history
- `frontend/` static chat page for Vercel
- Settings: chunk size 1200, overlap 150, top-k 3, model gpt-oss-120b
