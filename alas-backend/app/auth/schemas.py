from pydantic import BaseModel, EmailStr
from enum import Enum

class RoleEnum(str, Enum):
    admin = "admin"
    teacher = "teacher"
    student = "student"


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(BaseModel):
    email: EmailStr
    hashed_password: str
    role: str

class UserSignup(BaseModel):
    email: EmailStr
    password: str
    role: RoleEnum 
