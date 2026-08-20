import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_setting(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def get_gemini_api_key():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
    return api_key


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


def rewrite_query(query: str) -> str:
    """
    Rewrite user query into formal, search-friendly query
    """

    llm = ChatGoogleGenerativeAI(
        model=get_setting("GEMINI_GENERATION_MODEL", "gemini-3.5-flash"),
        api_key=get_gemini_api_key(),
    )

    prompt = f"""
Rewrite the query to improve retrieval.

Rules:
1. Keep it SHORT (max 8–10 words)
2. Preserve original meaning
3. Do NOT over-explain
4. Use terms likely present in documents
5. Avoid unnecessary adjectives

User Query:
{query}

Rewritten Query:
"""

    response = llm.invoke(prompt)
    return extract_text_content(response.content).strip()