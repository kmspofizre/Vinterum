from langchain_google_genai import ChatGoogleGenerativeAI
from config import Settings

def build_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    if not settings.gemini_api_key:
        raise RuntimeError("Set GOOGLE_API_KEY env var or pass api_key explicitly.")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
        temperature=0,
    )
