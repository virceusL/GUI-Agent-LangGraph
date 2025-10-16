from typing import Annotated, Dict, List, Literal, Optional, Tuple, cast
from typing_extensions import TypedDict
import requests

from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, RemoveMessage, ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig

from vlm_tools import VisionToolsConfig, UIMatchingTool
from peripheral_tools import MouseTool, KeyboardTool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# # parser
# from parser import parse_tool_call

from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
from dotenv import load_dotenv
load_dotenv()



# 节点的状态
class AgentState(TypedDict):
    """基于 MLLM, ReAct 模式, 定义核心状态

    ### 节点传递
    - `messages`: 消息历史, `list[dict]`
    - `user_task`: 全局目标
    - `screenshot_url`: 格式化后的 base64 编码图像, `observer` 节点填充。替换 `messages[-1].content` 中 `<image_pad>` 占位符
    - `screen_resolution`: 图像分辨率, `observer` 节点填充
    - `caption`: 图像描述, `agent` 节点填充
    ### 工具输出
    - `chosen_action`: 推理出的动作, `agent` 节点填充
    - `action_history`: 工具调用历史, `list[dict]` 形式
    
    ReAct, 只保留 `SystemMessage` + `HumanMessage` + `messages[-2:]`?
    """
    # 消息历史
    messages: Annotated[list, add_messages]
    locale: str
    user_task: str

    # 环境信息
    screenshot_url: Optional[str] # 格式化后的 base64 编码的屏幕截图
    screen_resolution: Tuple[int, int]

    # tool_call 相关信息
    action_history:  List[Dict] # 底层执行日志

# 获取 MLLM
def get_mllm():
    if os.getenv("OPENAI_API_BASE"):
        return ChatOpenAI(
            model=os.getenv("MODEL_ID", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0.7,
        )
    else:
        return ChatOpenAI(
            model=os.getenv("MODEL_ID", "gpt-3.5-turbo"),
            temperature=0.7,
        )

# 工具
ui_matching_tool = UIMatchingTool(VisionToolsConfig())
mouse_tool = MouseTool()
keyboard_tool = KeyboardTool()
tools = [ui_matching_tool, mouse_tool, keyboard_tool]

mllm = get_mllm()
mllm_tools = cast(BaseChatModel, mllm).bind_tools(tools)

def should_call_tools(state: AgentState) -> Literal["__end__", "tools"]:
    """判断是否需要调用工具"""
    last_message = state.get("messages", [])[-1]
    # If there is no function call, then we finish
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return "__end__"
    # Otherwise if there is, we call tools
    else:
        return "tools"

def capture_screen_base64() -> Tuple[str, Tuple[int, int]]:
    """获取屏幕截图 base64 编码"""
    try:
        from PIL import Image, ImageGrab
        import base64
        import io
        screenshot = ImageGrab.grab()
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")  # 也可以使用 JPEG 等其他格式
        screenshot_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        screenshot_url = f"data:image/png;base64,{screenshot_str}"
        return screenshot_url, screenshot.size
    except Exception as e:
        raise Exception(f"[ERROR][_Observer] Capturing screen: {e}")

env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

def executor_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Executor 节点, 负责推理和调用工具"""

    messages = state.get("messages", [])
    action_history = state.get("action_history", [])
    locale = state.get("locale", "en-US")

    # observe
    screenshot_url, screen_resolution = capture_screen_base64()

    if len(messages)==1 and isinstance(messages[-1], HumanMessage):
        user_task = messages[-1].content
        messages = [HumanMessage(
            content=[
                {"type": "image_url", "image_url": screenshot_url},
                {"type": "text", "text": user_task}
            ],
            id=messages[-1].id
        )]
    elif state.get("user_task", ""):
        user_task = state.get("user_task", "")
    else:
        raise Exception("No user task found")
    
    # execute
    EXECUTOR_PROMPT = env.get_template("executor.md").render({**state})
    ## parser 方式
    ## 加在 markdown 文件中
    # # Output Format
    #     Your words.
    #     ```xml
    #     <function=example_function_name>
    #     <parameter=example_parameter_1>value_1</parameter>
    #     <parameter=example_parameter_2>This is the value for the second parameter</parameter>
    #     </function>
    #     ```
    
    human_message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": screenshot_url},
            {"type": "text", "text": user_task}
        ],
        id=messages[0].id
    )
    if len(messages) > 5: # Human, AI, Tool, AI, Tool
        exec_prompt = [SystemMessage(content=EXECUTOR_PROMPT), human_message] + messages[-4:]
    else:
        exec_prompt = [SystemMessage(content=EXECUTOR_PROMPT), human_message] + messages[1:]
    
    response = mllm_tools.invoke(exec_prompt)

    if not response.tool_calls:
        goto = "__end__"
    else:
        goto = "tools"
        action_history.append(response.tool_calls)
    # if response.content:
    #     messages.append(AIMessage(content=response.content, name="executor"))
    # ai_message = parse_tool_call(response.content)
    # ai_message.name = "executor"
    return {
            "messages": response,
            "locale": locale,
            "user_task": user_task,
            "screenshot_url": screenshot_url,
            "screen_resolution": screen_resolution,
            "action_history": action_history,
        }

def build_base_graph():
    builder = StateGraph(AgentState)

    # TODO: PlanAct, planner node, reporter node
    builder.add_edge(START, "executor")
    builder.add_node("executor", executor_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge("tools", "executor")
    builder.add_conditional_edges(
        "executor",
        should_call_tools,
        ["tools", "__end__"]
    )

    return builder

def build_graph_with_memory():
    memory = MemorySaver()
    return build_base_graph().compile(checkpointer=memory)

def build_graph():
    return build_base_graph().compile()

def print_stream(graph, inputs, config):
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def run():
    graph = build_graph_with_memory()
    print(graph.get_graph(xray=True).draw_mermaid())
    
    config = {"configurable": {"thread_id": "gui_session"}}

    # 初始状态
    state = {
        "messages": [],
        "locale": "zh-CN",
        "task": "",
        "screenshot_url": "",
        "screen_resolution": (),
    }
    user_input = {"messages": {"role": "user", "content": f"{input('User>> ')}"}}

    try:
        print_stream(graph, user_input, config)
    except Exception as e:
        print("Error during agent invocation:", str(e))


if __name__ == "__main__":
    run()