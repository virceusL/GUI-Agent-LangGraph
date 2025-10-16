
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import base64

def encode_img(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
    
base64_image = encode_img("D:/31263/Pictures/desktop_720x480.png")
# img_url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
img_url = f"data:image/png;base64,{base64_image}"

import json
# 从 tools.json 导入工具描述, 储存为str
with open("tools.json", "r", encoding="utf-8") as f:
    tools_json = json.load(f)
tools_str = ""
for tool_json in tools_json:
    tool_str = json.dumps(tool_json, ensure_ascii=False)
    tools_str += f"{tool_str}\n"

from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, RemoveMessage, ToolMessage,
)
from jinja2 import Environment, FileSystemLoader, select_autoescape
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)
state = {
    "tools": tools_str,
    "locale": "zh_CN"
}
prompt = env.get_template("executor.md").render({**state})

user_prompt = "我想打开WeChat"

message=[
    SystemMessage(content=prompt),
    HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url": img_url
            },
        },
        {
            "type": "text", 
            "text": user_prompt
        },
    ])
]

from langchain_core.tools import tool
# from vlm_tools import VisionToolsConfig, UIMatchingTool
# from peripheral_tools import MouseTool, KeyboardTool

@tool
def ui_matching_tool():
    """ui_matching_tool, called for PRECISE match of UI element."""
    print("ui_matching_tool called")
    return "ui_matching_tool called"
@tool
def keyboard_tool():
    """keyboard_tool, called for actions like write, press, hotkey, sequence, etc."""
    print("keyboard_tool called")
    return "keyboard_tool called"
@tool
def mouse_tool():
    """mouse_tool, called for actions like click, drag, scroll, move cursor, etc."""
    print("mouse_tool called")
    return "mouse_tool called"


tools = [ui_matching_tool, mouse_tool, keyboard_tool]


from langchain_openai import ChatOpenAI
mmlm = ChatOpenAI(
    model=os.getenv("MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0.7,
).bind_tools(tools)
response = mmlm.invoke(message)
print(response.content)
print(response.tool_calls)
print("-"*30)
from parser import parse_tool_call
print(parse_tool_call(response.content))
