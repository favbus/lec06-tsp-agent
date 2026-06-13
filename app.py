from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from tsp_agent.agent import run_tsp_agent
from tsp_agent.models import City
from tsp_agent.parser import (
    cities_to_text,
    default_course_example,
    distance_matrix_course_example,
    parse_cities,
)


load_dotenv()

st.set_page_config(
    page_title="TSP 模拟退火优化智能体",
    page_icon="",
    layout="wide",
)


def random_cities(count: int, seed: int) -> list[City]:
    rng = random.Random(seed)
    return [
        City(f"C{i}", round(rng.uniform(0, 100), 2), round(rng.uniform(0, 100), 2))
        for i in range(count)
    ]


def route_figure(cities: list[City], route: list[int]) -> go.Figure:
    closed = route + [route[0]]
    xs = [cities[index].x for index in closed]
    ys = [cities[index].y for index in closed]
    labels = [cities[index].name for index in closed]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            marker={"size": 9, "color": "#dc2626"},
            line={"width": 2.5, "color": "#2563eb"},
            hovertemplate="%{text}: (%{x:.2f}, %{y:.2f})<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
    )
    return fig


def convergence_figure(solution: dict) -> go.Figure:
    history = solution["history"]
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["temperature_step"],
            y=df["current_length"],
            name="当前路线",
            mode="lines",
            line={"color": "#7c8aa0", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["temperature_step"],
            y=df["best_length"],
            name="历史最优",
            mode="lines",
            line={"color": "#0f766e", "width": 2.5},
        )
    )
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="温度轮次",
        yaxis_title="路线长度",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


st.title("基于模拟退火算法的旅行商问题 TSP 优化智能体")

with st.sidebar:
    st.header("输入设置")
    mode = st.radio(
        "城市数据",
        ["课件 4 城市示例", "随机生成", "手动输入"],
        index=1,
    )
    city_count = st.slider("随机城市数量", 4, 40, 12, disabled=mode != "随机生成")
    random_seed = st.number_input("随机种子", value=42, min_value=0, step=1)

    st.header("模拟退火参数")
    initial_temperature = st.number_input("初始温度 T0", value=1000.0, min_value=1.0, step=100.0)
    cooling_rate = st.slider("降温系数 alpha", 0.80, 0.99, 0.92, 0.01)
    chain_length = st.number_input("马尔可夫链长度 L", value=500, min_value=10, max_value=5000, step=50)
    final_temperature = st.number_input("终止温度 Tf", value=0.01, min_value=0.0001, step=0.01, format="%.4f")
    max_temperature_steps = st.number_input("最大温度轮次", value=220, min_value=10, max_value=1000, step=10)

if mode == "课件 4 城市示例":
    cities = default_course_example()
    default_text = cities_to_text(cities)
    distance_matrix = distance_matrix_course_example()
elif mode == "随机生成":
    cities = random_cities(city_count, int(random_seed))
    default_text = cities_to_text(cities)
    distance_matrix = None
else:
    default_text = "A,10,10\nB,80,20\nC,25,70\nD,90,85\nE,45,45"
    cities = []
    distance_matrix = None

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.subheader("城市坐标")
    input_text = st.text_area(
        "支持 CSV、JSON，或 A(10,20) 形式",
        value=default_text,
        height=260,
        label_visibility="collapsed",
    )
    user_request = st.text_area(
        "任务描述",
        value="请使用模拟退火算法求解旅行商问题，寻找一条总路程尽可能短的闭合路线，并输出详细求解报告。",
        height=120,
        label_visibility="collapsed",
    )
    run_button = st.button("运行智能体求解", type="primary", width="stretch")

with right:
    try:
        preview_cities = parse_cities(input_text)
        if mode == "课件 4 城市示例":
            st.caption("该模式使用课件中的 4x4 距离矩阵；坐标仅用于绘制路线示意图。")
        st.dataframe(
            pd.DataFrame([{"城市": c.name, "x": c.x, "y": c.y} for c in preview_cities]),
            width="stretch",
            hide_index=True,
        )
    except Exception as exc:
        preview_cities = []
        st.warning(str(exc))

if run_button:
    if not preview_cities:
        st.error("请先提供有效城市数据。")
        st.stop()

    params = {
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
        "chain_length": int(chain_length),
        "final_temperature": final_temperature,
        "max_temperature_steps": int(max_temperature_steps),
        "seed": int(random_seed),
    }

    with st.spinner("智能体正在规划、调用模拟退火工具并生成 PDF..."):
        matrix = distance_matrix if mode == "课件 4 城市示例" else None
        result = run_tsp_agent(
            user_request=user_request,
            cities=preview_cities,
            params=params,
            distance_matrix=matrix,
        )

    solution = result["solution"]
    st.success("求解完成")

    st.subheader("智能体流程")
    for item in result.get("logs", []):
        st.write(f"- {item}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("最优总路程", f"{solution['best_length']:.4f}")
    metric_cols[1].metric("总迭代次数", f"{solution['total_iterations']}")
    metric_cols[2].metric("接受坏解次数", f"{solution['worse_accepted']}")
    metric_cols[3].metric("改进比例", f"{solution['improvement_rate'] * 100:.2f}%")

    st.subheader("最优路线")
    st.code(" -> ".join(solution["closed_route_names"]), language="text")

    tabs = st.tabs(["路线图", "收敛曲线", "智能体规划", "结果解释", "PDF 报告"])
    with tabs[0]:
        st.plotly_chart(route_figure(preview_cities, solution["best_route"]), width="stretch")
    with tabs[1]:
        st.plotly_chart(convergence_figure(solution), width="stretch")
        st.dataframe(pd.DataFrame(solution["history"]).tail(20), width="stretch", hide_index=True)
    with tabs[2]:
        st.write(result.get("plan", ""))
    with tabs[3]:
        st.write(result.get("summary", ""))
        st.dataframe(pd.DataFrame(solution["sample_steps"]), width="stretch", hide_index=True)
    with tabs[4]:
        report_path = Path(result["report_path"])
        st.write(f"已生成：`{report_path}`")
        st.download_button(
            "下载 PDF 报告",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="application/pdf",
            width="stretch",
        )
else:
    st.info("设置城市数据和参数后，点击“运行智能体求解”。")
