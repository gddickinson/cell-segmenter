"""Start the Cell Segmenter web app on http://127.0.0.1:8146 (local only).

Run: python run_web.py   (install first: pip install -r requirements-web.txt)
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("cell_segmenter.web.app:app", host="127.0.0.1", port=8146)
