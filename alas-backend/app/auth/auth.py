from pydantic import BaseModel, EmailStr, Field
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional
from app.core.config import settings
from app.mongo import users_collection
from app.auth.utils import verify_password, pwd_context
from app.auth.schemas import UserLogin, UserSignup

# ------------------ Constants and Configurations ------------------


SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

router = APIRouter(tags=["Authentication"])

# ------------------ Helper Functions ------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT access token with an expiration time.
    
    Args:
        data (dict): Data to encode in the token (e.g., user email and role).
        expires_delta (Optional[timedelta]): Token expiration duration. Defaults to 30 minutes.
    
    Returns:
        str: Encoded JWT token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Extracts and validates the current user from the JWT token.
    
    Args:
        token (str): JWT token provided via OAuth2PasswordBearer.
    
    Returns:
        dict: Decoded token payload containing user information.
    
    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")


# ------------------ Routes ------------------

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserSignup):
    """
    Registers a new user in the system.
    """

    if users_collection is None:
        raise RuntimeError("users_collection is not initialized. Check your MongoDB setup.")

    print("MongoDB users_collection is ready.")


    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    hashed_password = pwd_context.hash(user.password)


    user_doc = {
        "email": user.email,
        "hashed_password": hashed_password,
        "role": user.role,
    }

    await users_collection.insert_one(user_doc)
    access_token = create_access_token({"sub": user.email, "role": user.role})
    return {"message": "User created successfully", "access_token": access_token, "token_type": "bearer"}

@router.post("/login")
async def login(user: UserLogin):
    """
    Authenticates a user and generates an access token.
    
    Args:
        user (UserLogin): User login details (email, password).
    
    Returns:
        dict: Access token and user role.
    
    Raises:
        HTTPException: If the email or password is incorrect.
    """

    user_in_db = await users_collection.find_one({"email": user.email})
    if not user_in_db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email or password")

    if not verify_password(user.password, user_in_db["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect password")

    access_token = create_access_token({"sub": user.email, "role": user_in_db["role"]})
    return {"access_token": access_token, "token_type": "bearer", "role": user_in_db["role"]}