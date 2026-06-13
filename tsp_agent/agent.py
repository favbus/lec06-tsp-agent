from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .models import AgentState, City
from .parser import city_from_dict
from .report import generate_pdf_report
from .tools import solve_tsp_by_simulated_annealing
from .visualization import save_convergence_chart, save_route_chart


def _city_list_text(cities: list[City]) -> str:
    return "\n".join(f"- {city.name}: ({city.x:.2f}, {city.y:.2f})" for city in cities)


def _secret_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else default
    except Exception:
        return default


def _get_llm() -> ChatOpenAI | None:
    load_dotenv(override=True)
    api_key = _secret_value("OPENAI_API_KEY")
    base_url = _secret_value("OPENAI_BASE_URL")
    model = _secret_value("OPENAI_MODEL", "gpt-5.5")
    if not api_key:
        return None
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        streaming=True,
        timeout=60,
        max_retries=2,
    )


def _ask_llm(system: str, user: str, fallback: str) -> str:
    llm = _get_llm()
    if llm is None:
        return fallback
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(response.content).strip() or fallback
    except Exception:
        return fallback


def plan_node(state: AgentState) -> AgentState:
    cities = [city_from_dict(item) for item in state["cities"]]
    params = state["params"]
    distance_matrix = state.get("distance_matrix")
    if distance_matrix:
        distance_text = (
            "本次路线长度必须使用用户给定的距离矩阵计算，不要写成欧氏距离。"
            f"距离矩阵：{distance_matrix}"
        )
    else:
        distance_text = "本次路线长度根据城市坐标的欧氏距离计算。"
    fallback = (
        "本次求解将 TSP 路线表示为城市排列，以闭合路线总长度作为目标函数。"
        "求解时先构造初始路线，再在每个温度下通过交换两个城市产生邻域解，"
        "使用 Metropolis 准则决定是否接受新解，最后输出最优路线、收敛过程和求解报告。"
    )
    plan = _ask_llm(
        "你是旅行商问题 TSP 的优化求解助手。请用中文给出简洁、严谨的求解规划。"
        "输出必须是纯文本，不要使用 Markdown 标题、加粗符号、代码块或 LaTeX 公式；公式用普通文本写法，例如 exp(-delta/T)。",
        (
            f"用户任务：{state.get('user_request', '求解 TSP')}\n"
            f"城市列表：\n{_city_list_text(cities)}\n"
            f"{distance_text}\n"
            f"模拟退火参数：{params}\n"
            "请说明建模方式、状态扰动、接受准则、降温方式和输出内容。"
            "只输出本次报告正文，不要提出后续服务建议。"
        ),
        fallback,
    )
    return {**state, "plan": plan, "logs": state.get("logs", []) + ["完成问题建模与求解规划。"]}


def solve_node(state: AgentState) -> AgentState:
    solution = solve_tsp_by_simulated_annealing.invoke(
        {
            "cities": state["cities"],
            "params": state["params"],
            "distance_matrix": state.get("distance_matrix"),
        }
    )
    return {**state, "solution": solution, "logs": state.get("logs", []) + ["调用模拟退火工具完成 TSP 求解。"]}


def summarize_node(state: AgentState) -> AgentState:
    solution = state["solution"]
    route_text = " -> ".join(solution["closed_route_names"])
    fallback = (
        f"模拟退火算法共执行 {solution['total_iterations']} 次邻域搜索，"
        f"得到最优闭合路线 {route_text}，"
        f"总路程为 {solution['best_length']:.4f}。"
        "高温阶段允许以一定概率接受较差路线，有助于跳出局部最优；随着温度降低，"
        "算法逐渐减少坏解接受概率并收敛到稳定路线。"
    )
    summary = _ask_llm(
        "你是优化算法报告助手。请根据求解结果生成中文结果分析，不要编造不存在的数据。"
        "输出必须是纯文本，不要使用 Markdown 标题、加粗符号、代码块或 LaTeX 公式；公式用普通文本写法，例如 exp(-delta/T)。",
        (
            f"规划：{state.get('plan', '')}\n"
            f"求解结果：最优路线={route_text}，"
            f"最优总路程={solution['best_length']:.4f}，"
            f"初始参考路程={solution['initial_length']:.4f}，"
            f"改进比例={solution['improvement_rate'] * 100:.2f}%，"
            f"总迭代次数={solution['total_iterations']}，"
            f"接受坏解次数={solution['worse_accepted']}。"
            "只输出本次报告正文，不要提出后续服务建议。"
        ),
        fallback,
    )
    return {**state, "summary": summary, "logs": state.get("logs", []) + ["完成结果解释。"]}


def report_node(state: AgentState) -> AgentState:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    cities = [city_from_dict(item) for item in state["cities"]]
    route_chart = save_route_chart(cities, state["solution"], output_dir / "tsp_best_route.png")
    convergence_chart = save_convergence_chart(state["solution"], output_dir / "tsp_convergence.png")
    report_path = generate_pdf_report(
        output_path=output_dir / "tsp_agent_report.pdf",
        user_request=state.get("user_request", ""),
        cities=cities,
        solution=state["solution"],
        plan=state.get("plan", ""),
        summary=state.get("summary", ""),
        convergence_chart=convergence_chart,
        route_chart=route_chart,
    )
    return {
        **state,
        "report_path": report_path,
        "chart_path": convergence_chart,
        "route_chart_path": route_chart,
        "logs": state.get("logs", []) + ["生成 PDF 求解报告。"],
    }


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("solve", solve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("report", report_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "solve")
    graph.add_edge("solve", "summarize")
    graph.add_edge("summarize", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_tsp_agent(
    *,
    user_request: str,
    cities: list[City],
    params: dict[str, Any],
    distance_matrix: list[list[float]] | None = None,
) -> AgentState:
    graph = build_agent_graph()
    initial_state: AgentState = {
        "user_request": user_request,
        "cities": [{"name": c.name, "x": c.x, "y": c.y} for c in cities],
        "params": params,
        "logs": ["收到用户输入，启动 TSP 优化求解流程。"],
    }
    if distance_matrix is not None:
        initial_state["distance_matrix"] = distance_matrix
    return graph.invoke(initial_state)
