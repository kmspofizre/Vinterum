from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    needs_retrieval: bool = Field(description="Should we query the course KB?")
    confidence: float = Field(ge=0.0, le=1.0)

class JudgeDecision(BaseModel):
    grounded: bool = Field(description="Answer supported by retrieved docs")
    relevant: bool = Field(description="Answer addresses the question")
    reason: str
