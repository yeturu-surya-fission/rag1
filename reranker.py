import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)


def extract_text_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    texts.append(item["content"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                texts.append(item.text)
        if texts:
            return "\n".join(texts)

    if hasattr(content, "text") and isinstance(content.text, str):
        return content.text

    return str(content)


def rerank_chunks(query, chunks):
    """
    Rerank chunks with one batched LLM scoring request.
    """

    if not chunks:
        return []

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
        api_key=api_key,
    )

    chunk_text = "\n\n".join(
        f"Chunk {index}:\n{chunk.page_content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    prompt = f"""
You are a relevance scoring system. Score every chunk for the query.

Scoring:
10 = directly answers the query
7 = highly relevant
5 = somewhat relevant
3 = weakly related
1 = irrelevant

Query:
{query}

{chunk_text}

Return exactly one integer score per chunk, in chunk order, separated by commas.
"""

    response = llm.invoke(prompt)
    content = extract_text_content(response.content)

    scores = [int(value) for value in re.findall(r"\b(?:10|[1-9])\b", str(content))]
    scores = (scores + [0] * len(chunks))[:len(chunks)]

    scored = list(zip(chunks, scores))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [c[0] for c in scored]