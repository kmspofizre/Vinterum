from langchain_core.prompts import ChatPromptTemplate

def router_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a router for a course assistant.\n"
         "Decide if we must retrieve from the course knowledge base.\n"
         "Retrieve if question is about course facts/definitions/policies/steps.\n"
         "No retrieve for smalltalk/general knowledge.\n"
         "Return JSON ONLY.\n{format_instructions}"),
        ("human", "Question: {question}")
    ])

def rag_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a course assistant. Use ONLY the provided context.\n"
         "If context is insufficient, say so and ask for clarification.\n"
         "Be concise and actionable."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

def direct_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer concisely."),
        ("human", "{question}")
    ])

def judge_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict evaluator.\n"
         "Given QUESTION, DOCUMENTS, ANSWER, decide grounded+relevant.\n"
         "Return JSON ONLY.\n{format_instructions}"),
        ("human",
         "QUESTION:\n{question}\n\nDOCUMENTS:\n{context}\n\nANSWER:\n{answer}")
    ])
