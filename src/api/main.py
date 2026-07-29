import os
import sys
from pathlib import Path

# 确保 src 目录在 sys.path 中, 使 from flow... 导入无缝解析
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import uvicorn

from src.api.app import app
from src.api.settings import Settings

if __name__ == "__main__":
    settings = Settings()
    # 默认单进程模式, 避免 Windows 下 SpawnProcess 产生孤儿僵尸进程; 显式设置 RELOAD=1 时开启热重载
    should_reload = os.environ.get("RELOAD", "0").lower() in ("1", "true")
    uvicorn.run(
        "src.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=should_reload,
        reload_dirs=[src_dir] if should_reload else None,
    )
