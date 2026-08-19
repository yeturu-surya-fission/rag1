from langchain_google_genai import ChatGoogleGenerativeAI


def rerank_chunks(query, chunks):
    """
    Rerank chunks using LLM scoring
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash"
    )

    scored = []

    for chunk in chunks:
        prompt = f"""
You are a relevance scoring system.

Score how relevant the following chunk is to the query.

Scoring:
10 = directly answers the query
7 = highly relevant
5 = somewhat relevant
3 = weakly related
1 = irrelevant

Query:
{query}

Chunk:
{chunk.page_content}

Score (only number):
"""

        response = llm.invoke(prompt)

        try:
            score = int(response.content.strip())
        except:
            score = 0

        scored.append((chunk, score))

    # sort descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # return only chunks
    return [c[0] for c in scored]