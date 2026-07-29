from fastapi import APIRouter, Depends

from src.api.deps import get_llm_key_status
from src.api.llm_key import LlmKeyStatus, _read_current_key, mask_key
from src.api.schemas import LlmStatusResponse

router = APIRouter()


@router.get("/status", response_model=LlmStatusResponse)
async def get_status(
    status: LlmKeyStatus = Depends(get_llm_key_status),
) -> LlmStatusResponse:
    key = _read_current_key()
    state = "red" if not key else status.state
    return LlmStatusResponse(
        state=state,
        masked_key=mask_key(key),
        last_error=status.last_error,
    )
