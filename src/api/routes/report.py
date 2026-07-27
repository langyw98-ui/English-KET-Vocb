import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from flow.ket_partner.db import Repos
from src.api.deps import User, get_current_user, get_db
from src.api.schemas import ReportCategoryResponse, ReportResponse, ReportWord

router = APIRouter()


@router.get("", response_model=ReportResponse)
async def report_counts(
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
) -> ReportResponse:
    repos = Repos.for_user(db, user.id)
    return ReportResponse(
        mastered_count=await repos.stats.count_by_category("mastered"),
        learning_count=await repos.stats.count_by_category("learning"),
        struggling_count=await repos.stats.count_by_category("struggling"),
        used_count=await repos.stats.count_by_category("used"),
        unused_count=await repos.stats.count_by_category("unused"),
        total_words=await repos.vocab.total_count(),
    )


@router.get("/{category}", response_model=ReportCategoryResponse)
async def report_by_category(
    category: str,
    page: int = 1,
    page_size: int = 100,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
) -> ReportCategoryResponse:
    valid = {"mastered", "learning", "struggling", "used", "unused"}
    if category not in valid:
        raise HTTPException(status_code=400, detail=f"invalid category: {category}")
    if page < 1 or page_size < 1 or page_size > 500:
        raise HTTPException(status_code=400, detail="invalid pagination params")

    repos = Repos.for_user(db, user.id)
    total = await repos.stats.count_by_category(category)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    rows = await repos.stats.list_by_category(category, offset=offset, limit=page_size)
    return ReportCategoryResponse(
        category=category,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        words=[ReportWord(**r) for r in rows],
    )
