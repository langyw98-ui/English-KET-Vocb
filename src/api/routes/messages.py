
import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from flow.ket_partner.db import Repos
from src.api.deps import User, get_current_user, get_db
from src.api.schemas import MessageOut

router = APIRouter()


@router.get("", response_model=list[MessageOut])
async def messages(
    limit: int = 15,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
) -> list[MessageOut]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be in [1, 100]")
    repos = Repos.for_user(db, user.id)
    rows = await repos.log.recent(limit=limit)
    return [MessageOut(**r) for r in rows]
