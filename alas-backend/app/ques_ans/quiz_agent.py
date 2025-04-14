from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

class QuizQuestionResponse(BaseModel):
    response: str
    question_id: int
    question: str
    options: dict
    difficulty: int
    topic: str


class QuizAnswerResponse(BaseModel):
    feedback: str
    next_question_id: int
    next_question: str
    next_options: dict
    next_difficulty: int
    next_topic: str


class AdaptiveJEEQuizAgent:
    def __init__(self, questions_df):
        self.questions = questions_df
        self.user_profile = {
            "current_difficulty": 1,  # Initial difficulty (1 = easiest)
            "performance_history": [],  # List of tuples: (question_id, is_correct)
            "weak_topics": set(),  # Topics where the user struggles
            "strength_topics": set(),  # Topics where the user excels
            "asked_questions": set()  # Questions already asked
        }

    def profile(self):
        """
        Returns the current user profile.
        """
        return {
            "current_difficulty": self.user_profile["current_difficulty"],
            "performance_history": self.user_profile["performance_history"],
            "weak_topics": list(self.user_profile["weak_topics"]),
            "strength_topics": list(self.user_profile["strength_topics"]),
            "asked_questions": list(self.user_profile["asked_questions"])
        }

    def select_question(self):
        """
        Selects a question based on the current difficulty and user profile.
        Returns:
            - Response message
            - Question ID
        """
        # Filter questions by difficulty and exclude already-asked questions
        filtered_questions = self.questions[
            (self.questions["difficulty"] == self.user_profile["current_difficulty"]) &
            (~self.questions.index.isin(self.user_profile["asked_questions"]))
        ]

        # Prioritize weak topics if available
        weak_topics = self.user_profile["weak_topics"]
        if weak_topics:
            filtered_questions = filtered_questions[filtered_questions["topic"].isin(weak_topics)]

        if filtered_questions.empty:
            raise ValueError(f"No questions available for difficulty: {self.user_profile['current_difficulty']}")

        next_qid = filtered_questions.sample(n=1).index[0]
        self.user_profile["asked_questions"].add(next_qid)  # Mark the question as asked
        response = f"Here's your next question at difficulty level {self.user_profile['current_difficulty']}."
        return response, next_qid

    def update_profile(self, question_id, is_correct):
        """
        Updates the user profile based on the user's performance.
        """
        # Update performance history
        self.user_profile["performance_history"].append((question_id, is_correct))

        # Get the topic of the current question
        topic = self.questions.loc[question_id]["topic"]

        # Update weak and strength topics
        if is_correct:
            self.user_profile["strength_topics"].add(topic)
            if topic in self.user_profile["weak_topics"]:
                self.user_profile["weak_topics"].remove(topic)
        else:
            self.user_profile["weak_topics"].add(topic)
            if topic in self.user_profile["strength_topics"]:
                self.user_profile["strength_topics"].remove(topic)

        # Update difficulty level
        if is_correct:
            self.user_profile["current_difficulty"] = min(self.user_profile["current_difficulty"] + 1, 5)  # Max difficulty is 5
        else:
            self.user_profile["current_difficulty"] = max(self.user_profile["current_difficulty"] - 1, 1)  # Min difficulty is 1


