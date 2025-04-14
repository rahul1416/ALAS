from fastapi import Depends, HTTPException
from .auth import get_current_user

def require_role(role: str):
    def role_checker(user: dict = Depends(get_current_user)):
        if user.get("role") != role:
            raise HTTPException(status_code=403, detail="Not authorized")
        return user
    return role_checker

admin_required = require_role("admin")
teacher_required = require_role("teacher")
student_required = require_role("student")
