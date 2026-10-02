# app/api/routers/pipeline_router.py

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.controller.scripts_controller import (
   start_full_pipeline_subprocesses
)
from app.utils.auth import require_admin_token

router = APIRouter(
    tags=["Scripts"],
    dependencies=[Depends(require_admin_token)],  # admin-only: requires the X-Admin-Token header
    responses={404: {"description": "Not found"}},
)

@router.post(
    "/run_pipeline",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger the full scrape → preprocess pipeline",
)
async def run_full_pipeline(background_tasks: BackgroundTasks):
    try:
       background_tasks.add_task(start_full_pipeline_subprocesses)
       return {"detail": "Full pipeline (subprocesses) triggered"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger pipeline: {e}"
        )
