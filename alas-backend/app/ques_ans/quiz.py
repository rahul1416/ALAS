from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ques_ans.quiz_agent import AdaptiveModel
from app.core.config import settings

router = APIRouter(prefix="/quiz", tags=["Quiz"])

QUESTIONS_CSV = settings.QUESTIONS
# Global cache for AdaptiveModel instances
adaptive_model_cache = {}

class AnswerRequest(BaseModel):
    user_id: str
    question_id: int
    user_answer: int

class StartRequest(BaseModel):
    user_id: str

async def get_adaptive_model(user_id: str):
    """
    Retrieve or create an AdaptiveModel instance for the given user.
    """
    if user_id not in adaptive_model_cache:
        agent = AdaptiveModel(QUESTIONS_CSV, user_id)
        # agent = AdaptiveModel(user_id)
        await agent.initialize()  # Initialize the model and profile asynchronously
        adaptive_model_cache[user_id] = agent
    return adaptive_model_cache[user_id]

@router.post("/start")
async def start_quiz(req: StartRequest):
    agent = await get_adaptive_model(req.user_id)
    question, qid = agent.select_question()
    if not question:
        raise HTTPException(status_code=404, detail="No questions available")
    return {
        "question_id": qid,
        "question": question["question"],
        "options": {
            1: question["option_1"],
            2: question["option_2"],
            3: question["option_3"],
            4: question["option_4"]
        },
        "difficulty": question["difficulty"],
        "topic": question["topic"]
    }

@router.post("/answer")
async def submit_answer(req: AnswerRequest):
    agent = await get_adaptive_model(req.user_id)
    try:
        question = agent.questions.loc[req.question_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="Question not found")

    correct_answer = int(question["correct_answer"])
    is_correct = req.user_answer == correct_answer

    agent.update_profile(req.question_id, is_correct)
    agent.online_train(question, is_correct)
    await agent.save_profile()

    # Get explanation from the LLM
    explanation = await agent.get_llm_response(
        question["question"],
        req.user_answer,
        correct_answer,
        {
            1: question["option_1"],
            2: question["option_2"],
            3: question["option_3"],
            4: question["option_4"]
        },
        question["difficulty"]
    )

    next_question, qid = agent.select_question()
    if next_question is None:
        return {"message": "No more questions available."}

    return {
        "correct": is_correct,
        "explanation": explanation,
        "next_question_id": qid,
        "next_question": next_question["question"],
        "next_options": {
            1: next_question["option_1"],
            2: next_question["option_2"],
            3: next_question["option_3"],
            4: next_question["option_4"]
        },
        "next_difficulty": next_question["difficulty"],
        "next_topic": next_question["topic"]
    }

@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    agent = await get_adaptive_model(user_id)
    return agent.profile