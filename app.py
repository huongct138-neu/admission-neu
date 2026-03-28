import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient

app = FastAPI(
    title="NEU Admission Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

API_PREFIX = "/api/admission_agent/v1"

BEARER_TOKEN = os.getenv("BEARER_TOKEN", "demo-secret-token")

OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "https://research.neu.edu.vn/ollama/api/embed")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:8b")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://research.neu.edu.vn/ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "admission_knowledge")

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.3"))

LAST_UPDATED = datetime.now(timezone.utc).isoformat()

METADATA = {
    "name": "Thông tin tuyển sinh",
    "description": "Agent hỗ trợ tư vấn tuyển sinh, phương thức xét tuyển, học phí và hồ sơ nhập học",
    "version": "1.0.0",
    "developer": "NEU AI Team",
    "capabilities": [
        "semantic_search",
        "answer_admission_questions",
        "guide_application_steps",
    ],
    "supported_models": [
        {
            "model_id": "qwen3-embedding:8b",
            "name": "Qwen3 Embedding 8B",
            "description": "Mô hình embedding dùng để truy xuất ngữ nghĩa",
            "accepted_file_types": ["pdf", "docx", "txt", "md"],
        }
    ],
    "sample_prompts": [
        "Các tổ hợp xét tuyển của ĐHKTQD?",
        "Các phương thức xét tuyển là gì?",
        "Học phí của các chương trình đào tạo tại ĐHKTQD?",
        "Thí sinh được đăng ký tối đa bao nhiêu nguyện vọng"
    ],
    "provided_data_types": [
        {
            "type": "qdrant_collection_info",
            "description": "Thông tin collection tuyển sinh đang dùng để tìm kiếm ngữ nghĩa",
        },
        {
            "type": "agent_config",
            "description": "Cấu hình cơ bản của admission agent",
        },
    ],
    "contact": "admission-support@neu.edu.vn",
    "status": "active",
}

SYSTEM_PROMPT = """Bạn là trợ lý tuyển sinh thông minh.

QUY TẮC:
- Chỉ trả lời dựa trên thông tin được cung cấp.
- Không suy đoán, không thêm kiến thức ngoài.
- Nếu không đủ thông tin, trả lời đúng 1 câu:
  "Theo các tài liệu được cung cấp, không có thông tin để trả lời câu hỏi này."

PHONG CÁCH TRẢ LỜI:
- Viết tự nhiên như ChatGPT
- Rõ ràng, dễ hiểu
- Ưu tiên chia thành các mục nếu phù hợp
- Có thể dùng:
  - tiêu đề (##)
  - bullet points (-)
- Không nhắc tới dữ liệu hay nguồn
"""

class HistoryItem(BaseModel):
    role: str
    content: str


class AskContext(BaseModel):
    language: Optional[str] = "vi"
    project: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    history: Optional[List[HistoryItem]] = None


class AskRequest(BaseModel):
    session_id: str
    model_id: str
    user: str
    prompt: str
    context: Optional[AskContext] = None


def verify_bearer_token(authorization: Optional[str]):
    expected = f"Bearer {BEARER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def embed_text(text: str) -> list[float]:
    payload = {
        "model": EMBED_MODEL,
        "input": text,
    }

    resp = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "embeddings" in data and data["embeddings"]:
        return data["embeddings"][0]

    if "embedding" in data and data["embedding"]:
        return data["embedding"]

    raise ValueError(f"Unexpected embedding response: {data}")


def search_qdrant(query_vector: list[float], limit: int = 5) -> list[dict]:
    client = get_qdrant_client()

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    points = response.points if hasattr(response, "points") else []

    results = []
    for point in points:
        payload = point.payload or {}
        score = float(getattr(point, "score", 0.0))

        if score < MIN_SCORE:
            continue

        results.append({
            "id": str(getattr(point, "id", "")),
            "score": score,
            "type": "document",
            "title": payload.get("source", "AI hỗ trợ tuyển sinh"),
            "content": payload.get("text", ""),
            "source": payload.get("source", ""),
        })

    return results


def build_context_block(docs: list[dict]) -> str:
    if not docs:
        return ""

    texts = []
    for doc in docs:
        content = doc.get("content", "").strip()
        if content:
            texts.append(content)

    return "\n\n".join(texts)

def call_llm(question: str, context_block: str) -> str:
    url = LLM_BASE_URL.rstrip("/") + "/api/chat"
    user_prompt = f"""Thông tin:

    {context_block}

    ---
    Câu hỏi: {question}

    Yêu cầu:
    - Trả lời tự nhiên như ChatGPT
    - Nếu có nhiều ý → chia bullet
    - Nếu phù hợp → thêm tiêu đề (##)
    - Không lặp lại dữ liệu thô
    - Không giải thích lan man

    Trả lời:"""
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()

    return resp.json()["message"]["content"]

@app.get(f"{API_PREFIX}/metadata")
def get_metadata():
    return METADATA


from fastapi import FastAPI, Header, HTTPException, Query
from typing import Optional

@app.get(f"{API_PREFIX}/data")
def get_data(
    type: str = Query(..., description="qdrant_collection_info | agent_config"),
    authorization: Optional[str] = Header(None, alias="Authorization"),):
    verify_bearer_token(authorization)
    if type == "qdrant_collection_info":
        try:
            client = get_qdrant_client()
            info = client.get_collection(QDRANT_COLLECTION)

            return {
                "status": "success",
                "data_type": type,
                "items": [
                    {
                        "collection_name": QDRANT_COLLECTION,
                        "host": QDRANT_HOST,
                        "port": QDRANT_PORT,
                        "config": str(info.config),
                    }
                ],
                "last_updated": LAST_UPDATED,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")

    if type == "agent_config":
        return {
            "status": "success",
            "data_type": type,
            "items": [
                {
                    "embed_url": OLLAMA_EMBED_URL,
                    "embed_model": EMBED_MODEL,
                    "top_k": TOP_K,
                    "min_score": MIN_SCORE,
                    "qdrant_collection": QDRANT_COLLECTION,
                }
            ],
            "last_updated": LAST_UPDATED,
        }

    raise HTTPException(status_code=404, detail=f"Data type '{type}' not found")


@app.post(f"{API_PREFIX}/ask")
@app.post(f"{API_PREFIX}/ask")
def ask_agent(req: AskRequest):
    start = time.time()

    try:
        # 1. Embed câu hỏi
        query_vector = embed_text(req.prompt)

        # 2. Search Qdrant
        docs = search_qdrant(query_vector, limit=TOP_K)

        # 3. Build context
        context_block = build_context_block(docs)

        # 4. Gọi LLM
        answer = call_llm(req.prompt, context_block)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask error: {str(e)}")

    return {
        "session_id": req.session_id,
        "status": "success",
        "content_markdown": answer,
        "meta": {
            "model": LLM_MODEL,
            "retrieved_count": len(docs),
        },
        "attachments": [],
    }


# def ask_agent(payload: AskRequest):
#     start = time.time()

#     try:
#         query_vector = embed_text(payload.prompt)
#         retrieved = search_qdrant(query_vector, limit=TOP_K)
#     except requests.HTTPError as e:
#         raise HTTPException(status_code=502, detail=f"Embedding service error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ask pipeline error: {str(e)}")

#     lines = ["## Trả lời", ""]
#     if retrieved:
#         lines.append(f"Bạn hỏi: **{payload.prompt}**")
#         lines.append("")
#         lines.append("Dưới đây là các nội dung gần nhất tìm được trong kho dữ liệu:")
#         lines.append("")

#         for i, item in enumerate(retrieved, start=1):
#             lines.append(f"### {i}. {item['title']}")
#             lines.append(item["content"] or "Không có nội dung tóm tắt.")
#             lines.append(f"- Điểm tương đồng: `{item['score']:.4f}`")
#             if item["source"]:
#                 lines.append(f"- Nguồn: `{item['source']}`")
#             lines.append("")
#     else:
#         lines.append(f"Chưa tìm thấy dữ liệu phù hợp cho câu hỏi: **{payload.prompt}**.")
#         lines.append("")
#         lines.append("## Gợi ý")
#         lines.append("- Hãy hỏi cụ thể hơn về hồ sơ nhập học, phương thức xét tuyển, ngành học hoặc học phí.")

#     attachments = []
#     if payload.context and payload.context.extra_data:
#         documents = payload.context.extra_data.get("document", [])
#         for doc_url in documents:
#             attachments.append({
#                 "type": "document",
#                 "url": doc_url,
#             })

#     elapsed_ms = int((time.time() - start) * 1000)

#     return {
#         "session_id": payload.session_id,
#         "status": "success",
#         "content_markdown": "\n".join(lines),
#         "meta": {
#             "model": payload.model_id,
#             "response_time_ms": elapsed_ms,
#             "tokens_used": 0,
#             "retrieved_count": len(retrieved),
#             "embed_model": EMBED_MODEL,
#             "qdrant_collection": QDRANT_COLLECTION,
#         },
#         "attachments": attachments,
#     }