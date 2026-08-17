"""
auth.py — JWT Authentication & Role-Based Access Control
=========================================================
Provides:
  - JWT token creation and verification
  - Role-based access: doctor | patient | admin
  - FastAPI dependency functions for protecting endpoints
  - Demo credentials (override in production via .env)

Usage in main.py:
    from auth import get_current_user, require_role, create_access_token, router as auth_router

    app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])

    # Protect an endpoint (any authenticated user):
    @app.get("/api/echonet/eda")
    async def echonet_eda(user = Depends(get_current_user)):
        ...

    # Restrict to doctors and admins:
    @app.post("/api/arthritis/train")
    async def train(user = Depends(require_role(["doctor", "admin"]))):
        ...

Demo credentials (change in production):
    admin  / admin123   → role: admin
    doctor / doctor123  → role: doctor
    patient/ patient123 → role: patient
"""

import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

try:
    from jose import JWTError, jwt
    _JOSE_AVAILABLE = True
except ImportError:
    _JOSE_AVAILABLE = False

# ── Password hashing: use bcrypt directly (bypasses passlib/bcrypt4.x conflict) ──
# passlib is intentionally NOT used here because passlib <1.7.5 is incompatible
# with bcrypt >=4.0 (removed __about__ attribute).  We call bcrypt directly.
try:
    import bcrypt as _bcrypt_lib

    def _hash(pw: str) -> str:
        """Hash password with bcrypt (bcrypt 4.x direct API)."""
        return _bcrypt_lib.hashpw(
            pw.encode("utf-8"), _bcrypt_lib.gensalt(rounds=12)
        ).decode("utf-8")

    def _verify(pw: str, hashed: str) -> bool:
        """Verify password against bcrypt hash."""
        try:
            return _bcrypt_lib.checkpw(pw.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False

    _BCRYPT_DIRECT = True

except ImportError:
    # Fallback: PBKDF2-SHA256 via stdlib hashlib (no third-party deps needed)
    _BCRYPT_DIRECT = False

    def _hash(pw: str) -> str:
        """Hash password with PBKDF2-SHA256 (fallback when bcrypt unavailable)."""
        salt = os.urandom(16).hex()
        h = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"),
                                 salt.encode(), 260_000).hex()
        return f"pbkdf2${salt}${h}"

    def _verify(pw: str, hashed: str) -> bool:
        """Verify PBKDF2 hash."""
        try:
            _, salt, stored_h = hashed.split("$", 2)
            computed = hashlib.pbkdf2_hmac(
                "sha256", pw.encode("utf-8"), salt.encode(), 260_000
            ).hex()
            return computed == stored_h
        except Exception:
            return False

from core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Settings (read from config if available)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from config import settings
    SECRET_KEY   = settings.secret_key
    ALGORITHM    = settings.algorithm
    EXPIRE_MINS  = settings.access_token_expire_minutes
except Exception:
    SECRET_KEY   = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    ALGORITHM    = "HS256"
    EXPIRE_MINS  = 60

# ──────────────────────────────────────────────────────────────────────────────
# In-memory user store (replace with DB in production)
# ──────────────────────────────────────────────────────────────────────────────

DEMO_USERS = {
    "admin":   {"username": "admin",   "role": "admin",   "hashed_password": _hash("admin123")},
    "doctor":  {"username": "doctor",  "role": "doctor",  "hashed_password": _hash("doctor123")},
    "patient": {"username": "patient", "role": "patient", "hashed_password": _hash("patient123")},
}

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    role:         str
    expires_in:   int  # seconds

class TokenData(BaseModel):
    username: Optional[str] = None
    role:     Optional[str] = None

class UserOut(BaseModel):
    username: str
    role:     str

# ──────────────────────────────────────────────────────────────────────────────
# JWT helpers
# ──────────────────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=EXPIRE_MINS))
    to_encode["exp"] = expire

    if not _JOSE_AVAILABLE:
        # Fallback: base64-encoded JSON (NOT secure — use only in dev)
        import base64, json
        payload = json.dumps(to_encode, default=str)
        return base64.urlsafe_b64encode(payload.encode()).decode()

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    if not _JOSE_AVAILABLE:
        import base64, json
        try:
            payload = json.loads(base64.urlsafe_b64decode(token + "=="))
            return TokenData(username=payload.get("sub"), role=payload.get("role"))
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role:     str = payload.get("role")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return TokenData(username=username, role=role)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI dependencies
# ──────────────────────────────────────────────────────────────────────────────

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> UserOut:
    """
    Dependency that validates the Bearer token.
    Returns the authenticated user or raises 401.
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_data = decode_token(token)
    user = DEMO_USERS.get(token_data.username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return UserOut(username=user["username"], role=user["role"])


def require_role(roles: List[str]):
    """
    Dependency factory that restricts an endpoint to users with specific roles.

    Usage:
        @app.post("/api/arthritis/train")
        async def train(user = Depends(require_role(["doctor", "admin"]))):
    """
    async def _check(user: UserOut = Depends(get_current_user)) -> UserOut:
        if user.role not in roles:
            logger.warning(
                "Authorization denied",
                extra={"username": user.username, "role": user.role, "required_roles": roles},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not permitted. Required: {roles}",
            )
        return user
    return _check


# Optional: dependency that allows unauthenticated access but attaches user if token provided
async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[UserOut]:
    if token is None:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Auth router
# ──────────────────────────────────────────────────────────────────────────────

router = APIRouter(tags=["Authentication"])

@router.post(
    "/token",
    response_model=Token,
    summary="Login and get JWT token",
    description=(
        "Submit username and password (form data) to receive a Bearer token. "
        "**Demo credentials:** admin/admin123, doctor/doctor123, patient/patient123"
    ),
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = DEMO_USERS.get(form_data.username)
    if not user or not _verify(form_data.password, user["hashed_password"]):
        logger.warning("Failed login attempt", extra={"username": form_data.username})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        {"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=EXPIRE_MINS),
    )
    logger.info("User logged in", extra={"username": user["username"], "role": user["role"]})
    return Token(
        access_token=token,
        role=user["role"],
        expires_in=EXPIRE_MINS * 60,
    )


@router.get("/me", response_model=UserOut, summary="Get current user info")
async def get_me(current_user: UserOut = Depends(get_current_user)):
    return current_user
