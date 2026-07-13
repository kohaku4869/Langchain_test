import os 
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langsmith import traceable

load_dotenv()

@tool
@traceable(name = "Tra cứu giá quả")
def get_fruit_price(query: str) -> str:
    """
    Nhận vào tên loại hoa quả bằng tiếng anh sẽ trả ra giá tiền của loại quả đấy
    """

    fruits = {"apple": 10, "grape":15, "kiwi":20, "mango":8, "orange":5}

    for fruit in fruits:
        if fruit in query:
            return str(fruits[fruit])
        
    return "The fruit does not exits"

@tool
@traceable(name = "Áp mã giảm giá")
def get_fruit_price_discount(fruit: str, price: str) -> str:
    """
    Nhận tên của quả và giá hiện tại của quả đó,Sử dụng sau khi đã có tên quả và giá của quá đó để nhận giá sau khi được giảm đã giảm
    """
    discount_price = {"apple": 0.9, "grape":0.8, "orange":0.7}

    for a in discount_price:
        if a in fruit:
            return str(int(price) * discount_price[a])
        
    return "The fruit is not discounted"

tools = [get_fruit_price,get_fruit_price_discount]
tool_map = {t.name:t for t in tools}


SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh. Bạn có quyền sử dụng các công cụ được cung cấp để giải quyết bài toán.
Hãy suy nghĩ từng bước. Khi đã thu thập đủ thông tin và có kết quả cuối cùng, hãy trả lời trực tiếp cho người dùng một cách rõ ràng."""

model = ChatNVIDIA(
  model="qwen/qwen3-next-80b-a3b-instruct",
  api_key= os.getenv("NVDIA_API_KEY"), 
  temperature=0,
  top_p=0.95,
  max_tokens=4096,
  reasoning = False
)
model_with_tools = model.bind_tools(tools)


@traceable(name = "W Agent")
def run_agent(query: str, max_step = 5):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    for step in range(max_step):

        #return AIMessage object
        response = model_with_tools.invoke(messages)
        messages.append(response)

        print ("="*30,"Step ",step+1,"="*30)

        if not response.tool_calls:
            print("Final Answer: ",response.content)
            return
        else:
            result = ""
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"] 
                tool_id = tool_call["id"]

                print(f"Running tool {tool_name} with tool_args {tool_args}")

                if (tool_name in tool_map):
                    executed_tool = tool_map[tool_name]
                    result = executed_tool.invoke(tool_args)
                else:
                    result = f"Không tìm thấy tool mang tên {tool_name}."
            print("Observation: ",result)
            messages.append(ToolMessage(content = result, tool_call_id = tool_id))
    
if __name__ == "__main__":
    run_agent("What does the price of the apple after discount")
