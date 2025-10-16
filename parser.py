import re
from langchain_core.messages import AIMessage
import uuid

def parse_tool_call(text: str) -> AIMessage:
    """
    Parse model output (pseudo-XML style) into LangChain AIMessage with tool_calls.
    """
    # content
    content_match = re.search(r"(.*?)```xml", text, re.DOTALL)
    if not content_match:
        raise ValueError("No <think>...</think> found in text")
    content = content_match.group(1).strip()

    # function name
    fn_match = re.search(r"<function=(.*?)>", text, re.DOTALL)
    if not fn_match:
        raise ValueError("No <function=...> found in text")
    fn_name = fn_match.group(1).strip()

    # parameters
    param_matches = re.findall(
        r"<parameter=(.*?)>(.*?)</parameter>", text, re.DOTALL
    )
    args = {}
    # 转换为int,array,tuple等
    for name, value in param_matches:
        if value.isdigit():
            value = int(value)
        elif value.startswith("[") and value.endswith("]"):
            if all(x.isdigit() for x in value.strip("[]").split(",")):
                value = [int(x) for x in value.strip("[]").split(",")]
            else:
                value = [x.strip() for x in value.strip("[]").split(",")]
        elif value.startswith("(") and value.endswith(")"):
            if all(x.isdigit() for x in value.strip("()").split(",")):
                value = tuple(int(x) for x in value.strip("()").split(","))
            else:
                value = tuple(x.strip() for x in value.strip("()").split(","))
            # value = tuple(int(x) for x in value.strip("()").split(","))
        args[name.strip()] = value.strip()

    ai_msg = AIMessage(
        content=content, # 保留原始输出
        tool_calls=[
            {
                "id": "call_" + "".join(str(uuid.uuid4()).split("-")),  # 可以生成唯一ID
                "name": fn_name,
                "args": args
            }
        ]
    )

    return ai_msg


if __name__ == "__main__":
    # ==== 测试 ====
    raw_model_output = """
    从屏幕截图来看，当前界面是电脑的桌面，为了打开 Chrome，我现在首先需要定位到 Chrome 在当前画面中的坐标。

    ```xml
    <function=ui_matching_tool>
    <parameter=query>Chrome</parameter>
    </function>
    ```
    """

    msg = parse_tool_call(raw_model_output)
    print(msg.content)
    print(msg.tool_calls)