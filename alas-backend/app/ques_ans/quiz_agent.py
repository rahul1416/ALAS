import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

from app.core.config import settings

class QuestionPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class AdaptiveModel:
    def __init__(self, questions_path, user_id, model_dir=settings.MODEL_DIR, user_db=settings.USER_PROFILE):
        self.questions = pd.read_csv(questions_path)
        self.questions["id"] = self.questions.index
        self.le_topic = LabelEncoder()
        self.questions["topic_encoded"] = self.le_topic.fit_transform(self.questions["topic"])
        self.input_dim = 4
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.user_db = user_db
        self.user_id = user_id
        self.profile = self.load_or_init_profile()
        self.model = self.load_model()

        # Groq client for LLM explanation
        self.groq_client = ChatGroq(temperature=0.7, model_name="llama3-8b-8192")
        self.adaptive_prompt = ChatPromptTemplate.from_template(
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

    def load_or_init_profile(self):
        if os.path.exists(self.user_db):
            with open(self.user_db) as f:
                profiles = json.load(f)
        else:
            profiles = {}

        if self.user_id not in profiles:
            profiles[self.user_id] = {
                "performance_history": [],
                "strength_topics": [],
                "weak_topics": [],
                "current_difficulty": 1
            }

        self.profiles = profiles
        return profiles[self.user_id]

    def save_profile(self):
        with open(self.user_db, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def extract_features(self, df):
        df = df.copy()
        df["topic_encoded"] = self.le_topic.transform(df["topic"])
        df["topic_match_weak"] = df["topic"].apply(lambda t: 1 if t in self.profile["weak_topics"] else 0)
        df["topic_match_strong"] = df["topic"].apply(lambda t: 1 if t in self.profile["strength_topics"] else 0)
        return df[["topic_encoded", "difficulty", "topic_match_weak", "topic_match_strong"]]

    def update_profile(self, qid, is_correct):
        question = self.questions.loc[qid]
        topic = question["topic"]
        self.profile["performance_history"].append((int(qid), is_correct))
        self.profile["performance_history"] = self.profile["performance_history"][-50:]

        if is_correct:
            self.profile["strength_topics"] = list(set(self.profile["strength_topics"] + [topic]))
            self.profile["weak_topics"] = [t for t in self.profile["weak_topics"] if t != topic]
            self.profile["current_difficulty"] = min(self.profile["current_difficulty"] + 1, 5)
        else:
            self.profile["weak_topics"] = list(set(self.profile["weak_topics"] + [topic]))
            self.profile["strength_topics"] = [t for t in self.profile["strength_topics"] if t != topic]
            self.profile["current_difficulty"] = max(self.profile["current_difficulty"] - 1, 1)

    def get_unanswered(self):
        return self.questions[~self.questions["id"].isin([q for q, _ in self.profile["performance_history"]])]

    def select_question(self):
        
        unanswered = self.get_unanswered().copy()

        if unanswered.empty:
            return None, None
        features = self.extract_features(unanswered).values.astype(np.float32)
        X_tensor = torch.tensor(features)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(X_tensor).squeeze().numpy()
        unanswered["score"] = scores
        best = unanswered.sort_values("score", ascending=False).iloc[0]
        return best.to_dict(), int(best["id"])

    def online_train(self, question, is_correct):
        topic_encoded = self.le_topic.transform([question["topic"]])[0]
        topic_match_weak = 1 if question["topic"] in self.profile["weak_topics"] else 0
        topic_match_strong = 1 if question["topic"] in self.profile["strength_topics"] else 0

        X = np.array([[topic_encoded, question["difficulty"], topic_match_weak, topic_match_strong]], dtype=np.float32)
        y = np.array([[int(is_correct)]], dtype=np.float32)

        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.model.train()
        optimizer.zero_grad()
        output = self.model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        self.save_model()

    def get_model_path(self):
        return os.path.join(self.model_dir, f"{self.user_id}_model.pt")

    def load_model(self):
        model = QuestionPredictor(self.input_dim)
        model_path = self.get_model_path()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        return model

    def save_model(self):
        torch.save(self.model.state_dict(), self.get_model_path())

    async def get_llm_response(self, question, user_answer, correct_answer, options, difficulty):
        prompt = self.adaptive_prompt.format(
            question=question,
            option_1=options[1],
            option_2=options[2],
            option_3=options[3],
            option_4=options[4],
            user_answer=user_answer,
            correct_answer=correct_answer,
            difficulty=difficulty,
            correct_option=options[correct_answer]
        )
        response = self.groq_client.invoke(prompt)
        return response.content


