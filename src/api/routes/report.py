import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from flow.ket_partner.config import load_config
from src.api.deps import User, get_current_user, get_db
from src.api.schemas import ReportCategoryResponse, ReportResponse, ReportWord
from src.persistence.repos import Repos
from src.reporting.ket_partner.categories import CATEGORIES, group_by_category

router = APIRouter()
_CFG = load_config()


@router.get("", response_model=ReportResponse)
async def report_counts(
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
) -> ReportResponse:
    """Spec §7 Option B: one SQL pull, Python classification."""
    repos = Repos.for_user(db, user.id)
    rows = await repos.stats.list_all_with_vocab()
    bucket = group_by_category(rows, _CFG)
    return ReportResponse(
        mastered_count=len(bucket["mastered"]),
        learning_count=len(bucket["learning"]),
        struggling_count=len(bucket["struggling"]),
        used_count=len(bucket["used"]),
        unused_count=len(bucket["unused"]),
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
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"invalid category: {category}")
    if page < 1 or page_size < 1 or page_size > 500:
        raise HTTPException(status_code=400, detail="invalid pagination params")

    repos = Repos.for_user(db, user.id)
    rows = await repos.stats.list_all_with_vocab()
    bucket = group_by_category(rows, _CFG)
    category_rows = bucket[category]
    total = len(category_rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_rows = category_rows[offset:offset + page_size]
    return ReportCategoryResponse(
        category=category,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        words=[ReportWord(**r) for r in page_rows],
    )
