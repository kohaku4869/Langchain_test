import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import time
import json
from langsmith import traceable
from langsmith.wrappers import wrap_openai

load_dotenv()
client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.getenv("NVDIA_API_KEY")
)
client = wrap_openai(client)

@traceable(run_type = "tool", name = "Tra cứu giá quả")
def get_fruit_price(query: str) -> str:
    """
    Get fruit price by fruit name
    """

    fruits = {"apple": 10, "grape":15, "kiwi":20, "mango":8, "orange":5}

    for fruit in fruits:
        if fruit in query:
            return str(fruits[fruit])
        
    return "The fruit does not exits"

@traceable(run_type = "tool", name = "Áp mã giảm giá")
def get_fruit_price_discount(fruit: str, price: str) -> str:
    """
    Get the fruit's price after discount
    """
    discount_price = {"apple": 0.9, "grape":0.8, "orange":0.7}

    for a in discount_price:
        if a in fruit:
            return str(int(price) * discount_price[a])
        
    return "The fruit is not discounted"

AVAILABLE_TOOLS = {
    "Get_fruit_price": get_fruit_price,
    "Get_fruit_price_discount": get_fruit_price_discount
}

#ReAct prompt

SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh. Bạn PHẢI giải quyết bài toán của người dùng bằng cách suy nghĩ logic và sử dụng các công cụ được cung cấp theo đúng cấu trúc dưới đây.

Bạn có quyền truy cập vào các công cụ sau:
- Get_fruit_price: Nhận vào tên loại hoa quả bằng tiếng anh sẽ trả ra giá tiền của loại quả đấy
- Get_fruit_price_discount:Nhận tên của quả và giá hiện tại của quả đó,Sử dụng sau khi đã có tên quả và giá của quá đó để nhận giá sau khi được giảm đã giảm

Các công cụ nhận đầu vào và trả ra đầu ra như sau:
- Get_fruit_price: def get_fruit_price(query: str) -> str
- Get_fruit_price_discount: def get_fruit_price_discount(fruit: str, price: str) -> str

Quy trình bạn bắt buộc phải tuân theo ở mỗi bước suy nghĩ:

Thought: Ghi lại suy nghĩ hiện tại của bạn, bạn cần biết thêm điều gì và nên chọn công cụ nào tiếp theo.
Action: Tên công cụ bạn chọn (chỉ được chọn chính xác 1 trong 2 tên: Get_fruit_price hoặc Get_fruit_price_discount).
Action Input: Tham số đầu vào duy nhất truyền cho công cụ đó. Luôn trả về dưới dạng json

Sau khi bạn đưa ra Action và Action Input, hệ thống sẽ tự động chạy công cụ và trả về kết quả dưới dạng:
Observation: [Kết quả từ công cụ]

Quy trình trên sẽ lặp lại cho đến khi bạn tích lũy đủ thông tin. Khi đã tìm ra câu trả lời chính xác cuối cùng, bạn hãy dừng lại và trả về kết quả theo cấu trúc:
Thought: Tôi đã có câu trả lời cuối cùng.
Final Answer: [Câu trả lời hoàn chỉnh, chi tiết dành cho người dùng]

Bắt đầu nhiệm vụ!
"""

@traceable(run_type = "chain",name = "W/O Agent")
def run_react_agent(user_question:str, max_steps = 5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    for step in range(max_steps):
        print("="*15,"STEP ",step+1,"="*15)

        response = client.chat.completions.create(
        model="qwen/qwen3-next-80b-a3b-instruct",
        messages=messages,
        temperature=0,
        top_p=0.95,
        extra_body={"chat_template_kwargs":{"thinking":False}},
        stream=False,
        stop=["Observation:", "\nObservation:"]
        )
        

        
        llm_output = response.choices[0].message.content
        print(llm_output)

        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            print("="*30)
            print(f"[KẾT QUẢ CUỐI CÙNG]: {final_answer}")
            print("="*30)
            return


        action = re.search(r"Action:\s*(\w+)", llm_output)
        action_input = re.search(r"Action Input:\s*(.*)", llm_output)

        if action and action_input:
            tool_name = action.group(1).strip()
            tool_args = json.loads(action_input.group(1).strip())

            print(f"Running tool {tool_name} with tool_args {tool_args}")

            if tool_name in AVAILABLE_TOOLS:
                tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                print("Observation: ",tool_result)

                messages.append({"role": "user", "content": f"Observation: {tool_result}"})
            else:
                error_msg = f"The tool {tool_name} does not exits"
                print("Observation: ",error_msg)
                messages.append({"role": "user", "content": f"Observation: {error_msg}"})

        else:
            error_msg = "Error: Can not find Action/Action Input correctly, please try again"
            print(f"Observation: {error_msg}\n")
            messages.append({"role": "user", "content": f"Observation: {error_msg}"})
        time.sleep(3)

    print("Agent stops because it reachs max step")

if __name__ == "__main__":
    run_react_agent("What does the price of the apple after discount")