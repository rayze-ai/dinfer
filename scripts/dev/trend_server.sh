#!/usr/bin/env bash
# 性能趋势面板：静态服务 test/qwen3/（perf-trend.html 每 10 s 重读 perf-trend.csv）。
#   bash scripts/dev/trend_server.sh [PORT] [HOST]     # 默认 8090、0.0.0.0；打开 http://<host>:8090/perf-trend.html
# 每次 bench_compare.sh 跑完会重生成 CSV/HTML，浏览器里自动看到新一轮。只读静态文件，无鉴权；别暴露到公网。
set -euo pipefail
cd "$(dirname "$0")/../../test"   # 服务 test/ 根：index.html 总览 + qwen3/ qwen35/ 各自 perf-trend.html（之前只服务 test/qwen3，index.html 404）
PORT="${1:-8090}"; HOST="${2:-0.0.0.0}"
python3 -I - "$PORT" "$HOST" <<'PY'
import http.server, sys, socketserver
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store"); super().end_headers()
    def log_message(self, *a): pass
port, host = int(sys.argv[1]), sys.argv[2]
print(f"trend dashboard: http://{host}:{port}/index.html  (qwen3/perf-trend.html, qwen35/perf-trend.html)", flush=True)
socketserver.TCPServer.allow_reuse_address = True
http.server.ThreadingHTTPServer((host, port), H).serve_forever()
PY
