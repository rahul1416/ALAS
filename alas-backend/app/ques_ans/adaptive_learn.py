import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class AdaptiveJEEQuizAgent:
    def __init__(self, question_file, llm_model="llama3-70b-8192"):
        self.questions = pd.read_csv(question_file)
        self.llm = ChatGroq(model_name=llm_model, temperature=0.7)
        self.user_profile = {
            "current_difficulty": 1,
            "performance_history": [],
            "weak_topics": set(),
            "strength_topics": set(),
            "asked_questions": set()
        }
        
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

    def update_profile(self, question_id, is_correct):
        question = self.questions.loc[question_id]
        topic = question['topic']
        difficulty = question['difficulty']
        
        self.user_profile["asked_questions"].add(question_id)
        self.user_profile["performance_history"].append({
            "question_id": question_id,
            "is_correct": is_correct,
            "difficulty": difficulty,
            "topic": topic
        })
        
        if is_correct:
            self.user_profile["strength_topics"].add(topic)
            if topic in self.user_profile["weak_topics"]:
                self.user_profile["weak_topics"].remove(topic)
        else:
            self.user_profile["weak_topics"].add(topic)
            if topic in self.user_profile["strength_topics"]:
                self.user_profile["strength_topics"].remove(topic)
        
        last_5 = [r["is_correct"] for r in self.user_profile["performance_history"][-5:]]
        if len(last_5) >= 3:
            success_rate = sum(last_5)/len(last_5)
            if success_rate > 0.7:
                self.user_profile["current_difficulty"] = min(5, self.user_profile["current_difficulty"] + 0.5)
            elif success_rate < 0.3:
                self.user_profile["current_difficulty"] = max(1, self.user_profile["current_difficulty"] - 0.5)

    def select_question(self, prev_question_id=None, user_answer=None):
        if prev_question_id is not None and user_answer is not None:
            prev_question = self.questions.loc[prev_question_id]
            is_correct = str(user_answer) == str(prev_question['correct_answer'])
            self.update_profile(prev_question_id, is_correct)
            
            prompt = self.adaptive_prompt.format(
                question=prev_question['question'],
                option_1=prev_question['option_1'],
                option_2=prev_question['option_2'],
                option_3=prev_question['option_3'],
                option_4=prev_question['option_4'],
                user_answer=user_answer,
                correct_answer=prev_question['correct_answer'],
                difficulty=prev_question['difficulty'],
                correct_option=prev_question[f"option_{prev_question['correct_answer']}"]
            )
            llm_response = self.llm.invoke(prompt).content
            
            next_question = self._find_next_question(
                self.user_profile["current_difficulty"],
                prev_question['topic'] if is_correct else None
            )
            return llm_response, next_question.name
        else:
            next_question = self._find_next_question(1)
            return "Welcome to Adaptive JEE Quiz! Let's begin.", next_question.name

    def _find_next_question(self, target_diff, target_topic=None):
        candidates = self.questions[
            (self.questions['difficulty'] >= target_diff - 1) & 
            (self.questions['difficulty'] <= target_diff + 1) &
            (~self.questions.index.isin(self.user_profile["asked_questions"]))
        ]
        
        if candidates.empty:
            candidates = self.questions[~self.questions.index.isin(self.user_profile["asked_questions"])]
        
        if len(self.user_profile["weak_topics"]) > 0:
            weak_candidates = candidates[candidates['topic'].isin(self.user_profile["weak_topics"])]
            if not weak_candidates.empty:
                candidates = weak_candidates
        elif target_topic:
            topic_candidates = candidates[candidates['topic'] == target_topic]
            if not topic_candidates.empty:
                candidates = topic_candidates
        
        if candidates.empty:
            self.user_profile["asked_questions"] = set()
            candidates = self.questions
        
        candidates['weight'] = 1/(1 + abs(candidates['difficulty'] - target_diff))
        return candidates.sample(1, weights='weight').iloc[0]
    
    def profile(self):
        return self.user_profile

def run_quiz(question_file):
    quiz = AdaptiveJEEQuizAgent(question_file)
    prev_question_id = None

    while True:
        if prev_question_id is None:
            response, next_qid = quiz.select_question()
            print(response)
        else:
            user_answer = input("\nYour Answer (1/2/3/4 or 'exit' to quit): ")
            if user_answer.lower() == "exit":
                print("\nThank you for playing! Keep learning.")
                break
            
            try:
                user_answer = int(user_answer)
                if not 1 <= user_answer <= 4:
                    raise ValueError
            except:
                print("Please enter a number between 1-4")
                continue
            
            response, next_qid = quiz.select_question(prev_question_id, user_answer)
            print("\n" + response)  # Only print the LLM's response
        
        # Show next question (only one at a time)
        prev_question_id = next_qid
        current_question = quiz.questions.loc[prev_question_id]
        print(f"\nQuestion (Difficulty: {current_question['difficulty']}):")
        print(current_question['question'])
        print(f"1) {current_question['option_1']}")
        print(f"2) {current_question['option_2']}")
        print(f"3) {current_question['option_3']}")
        print(f"4) {current_question['option_4']}")

    print(quiz.profile())
if __name__ == "__main__":
    run_quiz("corrected_questions.csv")
    