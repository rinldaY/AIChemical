"""
scripts/serve_dashboard.py
--------------------------
รัน Local Web Server เพื่อเปิดหน้า Dashboard แสดงผลลัพธ์ของโปรเจกต์
ใช้งานง่าย: python scripts/serve_dashboard.py
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8080
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)


def main():
    url = f"http://localhost:{PORT}/index.html"
    print("=" * 70)
    print("AI-Based Chemical Reactor Anomaly Detection Web Dashboard")
    print("=" * 70)
    print(f"กำลังเปิด Dashboard ที่: {url}")
    print("กด Ctrl+C เพื่อหยุดการทำงานของ Server")
    print("=" * 70)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nปิดการทำงานของ Web Server เรียบร้อยแล้ว")


if __name__ == "__main__":
    main()
