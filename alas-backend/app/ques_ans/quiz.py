from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ques_ans.quiz_agent import AdaptiveJEEQuizAgent, QuizQuestionResponse, QuizAnswerResponse
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd

router = APIRouter(prefix="/quiz", tags=["Quiz"])

QUESTIONS = "app/db/corrected_questions.csv"
question = pd.read_csv(QUESTIONS)
quiz_agent = AdaptiveJEEQuizAgent(question)

class AnswerRequest(BaseModel):
    prev_question_id: int
    user_answer: int

# Adaptive Feedback Prompt
adaptive_prompt = ChatPromptTemplate.from_template(
    """You are an intelligent JEE quiz system. When the user answers:
    
    1. For correct answers:
    - Say "Correct! 🎉" 
    - Give a brief explanation
    - Don't suggest another question (the system will handle that)

    2. For incorrect answers:
    - Say "Incorrect. The answer is {correct_option}."
    - Give a brief explanation
    - Don't suggest another question

    Current Question: {question}
    Options:
    1) {option_1}
    2) {option_2}
    3) {option_3}
    4) {option_4}

    User's Answer: {user_answer}
    Correct Answer: {correct_answer}
    Difficulty: {difficulty}

    Provide only the response to the user's answer:"""
)

groq_client = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")


@router.get("/start", response_model=QuizQuestionResponse)
async def start_quiz():
    """
    Start a new adaptive quiz session.
    
    Returns:
        - First question with options
        - Welcome message
        - Initial difficulty level
    """
    try:
        response, next_qid = quiz_agent.select_question()
        current = quiz_agent.questions.loc[next_qid]

        return {
            "response": response,
            "question_id": next_qid,
            "question": current["question"],
            "options": {
                1: current["option_1"],
                2: current["option_2"],
                3: current["option_3"],
                4: current["option_4"]
            },
            "difficulty": current["difficulty"],
            "topic": current["topic"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start quiz: {str(e)}")

@router.post("/answer", response_model=QuizAnswerResponse)
async def next_question(question_id: int, user_answer: int):
    """
    Process the user's answer and return the next question with feedback.
    
    Args:
        - question_id: ID of the current question
        - user_answer: User's selected option (1-4)
    
    Returns:
        - Feedback on the user's answer
        - Next question details
    """
    try:
        # Get the current question
        current = quiz_agent.questions.loc[question_id]
        correct_answer = int(current["correct_answer"])
        is_correct = user_answer == correct_answer

        # Update the user profile
        quiz_agent.update_profile(question_id, is_correct)

        # Generate feedback using the language model
        prompt = adaptive_prompt.format(
            question=current["question"],
            option_1=current["option_1"],
            option_2=current["option_2"],
            option_3=current["option_3"],
            option_4=current["option_4"],
            user_answer=user_answer,
            correct_answer=correct_answer,
            correct_option=current[f"option_{correct_answer}"],
            difficulty=current["difficulty"]
        )
        feedback_response = groq_client.invoke(prompt)

        # Select the next question
        response, next_qid = quiz_agent.select_question()
        next_question_data = quiz_agent.questions.loc[next_qid]

        return {
            "feedback": feedback_response.content,
            "next_question_id": next_qid,
            "next_question": next_question_data["question"],
            "next_options": {
                1: next_question_data["option_1"],
                2: next_question_data["option_2"],
                3: next_question_data["option_3"],
                4: next_question_data["option_4"]
            },
            "next_difficulty": next_question_data["difficulty"],
            "next_topic": next_question_data["topic"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")


@router.get("/profile")
async def get_user_profile():
    """
    Get the current user profile including:
    - Performance history
    - Weak/strong topics
    - Current difficulty level
    - Asked questions
    """
    try:
        return quiz_agent.profile()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/question/{question_id}")
async def get_specific_question(question_id: int):
    """
    Get a specific question by ID
    
    Args:
        - question_id: ID of the question to retrieve
    """
    try:
        question = quiz_agent.questions.loc[question_id]
        return {
            "question_id": question_id,
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
    except KeyError:
        raise HTTPException(status_code=404, detail="Question not found")
