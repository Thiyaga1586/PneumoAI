import os
import uvicorn

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("RELOAD", "0") == "1"

    uvicorn.run(
        "pneumoai.serving.api.app:app",
        host=host,
        port=port,
        reload=reload_flag,
    )

if __name__ == "__main__":
    main()
