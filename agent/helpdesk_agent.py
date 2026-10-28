import os
import re
from openai import OpenAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic
class HelpDeskResponse(BaseModel):
    answers: str = Field(
        description="The detailed resolution to the user's issue based exclusivley on the context"
    )
    cited_chunk_ids: list[int] = Field(
        description="A list of the exact document IDs (e.g. 47, 55, 76, etc) from full context"
    )
    is_answerable: bool = Field(
        description="True if the answer is included in the context. False if the answer is not found in the context."
    )


def generate_response(user_query,full_context):
    system_prompt = """You are a strict Level 1 IT Helpdesk agent. 
                You must answer the user's question explicitly and EXCLUSIVELY using the provided Context Documents. 
                
                CRITICAL RULES:
                1. DO NOT use your outside knowledge, general IT knowledge, or memory.
                2. If the context mentions the user's topic but does not fully define or solve it, explain exactly what the context says and stop. Do not fill in the blanks.
                3. Do not invent features, security concepts, or steps that are not explicitly written in the source text.
                4. If the context does not contained in the context, you must set 'is_answerable' to False, and 'True' if the correrct answer is found in the context.
                """
    user_prompt = f"Context documents: {full_context}\n User Issue: {user_query}"
    response = client.chat.completions.parse(
         model='gpt-4o-mini',
        messages=[
            {"role":"developer","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        response_format=HelpDeskResponse
    )
    return response.choices[0].message.parsed