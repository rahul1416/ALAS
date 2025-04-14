from fastapi import APIRouter, Depends
from app.auth.deps import admin_required

router = APIRouter(prefix="/student", tags=["Admin"])

@router.get("/dashboard")
def admin_dashboard(user = Depends(admin_required)):
    return {"message": f"Welcome Admin {user['username']}"}
