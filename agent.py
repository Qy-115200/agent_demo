import os
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env
load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)
MODEL = os.getenv("MODEL", "deepseek-chat")

# 沙箱目录：Agent 只能在这里操作
ROOT = (Path(__file__).parent / "sandbox").resolve()
ROOT.mkdir(exist_ok=True)


def safe_path(path: str) -> Path:
    p = (ROOT / path).resolve()
    if not str(p).startswith(str(ROOT)):
        raise ValueError("路径越界，只允许操作 sandbox 目录")
    return p


def read_file(path: str) -> str:
    return safe_path(path).read_text(encoding="utf-8")[:4000]


def write_file(path: str, content: str) -> str:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {path}"


def bash(cmd: str) -> str:
    forbidden = ["rm -rf", "sudo", "curl", "wget", "format", "del /f"]
    if any(x in cmd for x in forbidden):
        return "命令被安全策略拒绝"
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=30, cwd=ROOT,
        )
        return (r.stdout + r.stderr)[:4000]
    except subprocess.TimeoutExpired:
        return "命令超时"


tool_map = {
    "read_file": read_file,
    "write_file": write_file,
    "bash": bash,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取 sandbox 内文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件到 sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "在 sandbox 目录执行 shell 命令",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
]

messages = [
    {
        "role": "system",
        "content": (
            "你是一个编码代理，只能在 sandbox 目录内工作。"
            "需要时调用工具，完成后直接回答。"
        ),
    }
]

task = input("请输入任务：")
messages.append({"role": "user", "content": task})

for step in range(10):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    msg = resp.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print("\nAgent:", msg.content)
        break

    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        print(f"[工具] {name} {args}")
        try:
            result = tool_map[name](**args)
        except Exception as e:
            result = f"错误: {e}"
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })
else:
    print("达到最大步数，停止。")