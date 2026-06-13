from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models import City


def _styles() -> dict[str, ParagraphStyle]:
    registerFont(UnicodeCIDFont("STSong-Light"))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ChineseTitle",
            parent=base["Title"],
            fontName="STSong-Light",
            fontSize=22,
            leading=28,
            alignment=1,
            spaceAfter=18,
        ),
        "h2": ParagraphStyle(
            "ChineseH2",
            parent=base["Heading2"],
            fontName="STSong-Light",
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "ChineseBody",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=10.5,
            leading=16,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "ChineseSmall",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=9,
            leading=13,
            spaceAfter=5,
        ),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    safe = html.escape(_clean_report_text(text)).replace("\n", "<br/>")
    return Paragraph(safe, style)


def _clean_report_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    in_code_block = False
    skip_rest = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if any(
            phrase in line
            for phrase in (
                "如果你需要",
                "我可以继续",
                "我还可以继续",
                "继续帮你",
                "课程作业格式",
            )
        ):
            skip_rest = True
        if skip_rest:
            continue
        if not in_code_block and re.fullmatch(r"[-*_]{3,}", line):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", raw_line)
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    paragraphs = re.split(r"\n\s*\n", text)
    blocked_phrases = (
        "提示词工程",
        "轻量工具调用",
        "公开大模型",
        "模型调用失败",
        "Service Unavailable",
        "server_error",
        "智能体求解流程",
        "作业",
    )
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if not any(phrase in paragraph for phrase in blocked_phrases)
    ]
    text = "\n\n".join(paragraphs)

    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\\+\[|\\+\]|\\+\(|\\+\)", "", text)
    text = re.sub(r"\\+text\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\+frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)

    replacements = {
        r"\\+Delta\b": "Delta",
        r"\\+delta\b": "delta",
        r"\\+pi\b": "pi",
        r"\\+alpha\b": "alpha",
        r"\\+beta\b": "beta",
        r"\\+exp\b": "exp",
        r"\\+left\b": "",
        r"\\+right\b": "",
        r"\\+leq?\b": "<=",
        r"\\+geq?\b": ">=",
        r"\\+times\b": "*",
        r"\\+to\b": "->",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\\+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table(data: list[list[Any]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5f0ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172033")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def generate_pdf_report(
    *,
    output_path: str | Path,
    user_request: str,
    cities: list[City],
    solution: dict[str, Any],
    plan: str,
    summary: str,
    convergence_chart: str,
    route_chart: str,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="TSP 模拟退火求解报告",
    )
    story = []
    story.append(_p("基于模拟退火算法的 TSP 优化求解报告", styles["title"]))
    story.append(_p(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["small"]))
    story.append(_p(f"用户任务：{user_request or '求解旅行商问题，寻找总路程尽可能短的闭合路线。'}", styles["body"]))

    story.append(_p("一、问题建模", styles["h2"]))
    story.append(
        _p(
            "旅行商问题要求从某个城市出发，访问每个城市恰好一次，最后回到出发城市。"
            "本次求解将一个路线编码为城市编号的排列，目标函数为闭合路线的总距离，优化目标是最小化该距离。",
            styles["body"],
        )
    )
    city_rows = [["城市", "x", "y"]] + [[c.name, f"{c.x:.3f}", f"{c.y:.3f}"] for c in cities]
    story.append(_table(city_rows, [4 * cm, 4 * cm, 4 * cm]))
    if solution.get("distance_mode") == "distance_matrix" and solution.get("distance_matrix"):
        story.append(
            _p(
                "本次求解使用用户给定的距离矩阵计算路线长度；上表坐标仅用于页面和报告中的路线示意图。",
                styles["body"],
            )
        )
        matrix = solution["distance_matrix"]
        matrix_rows = [[""] + [city.name for city in cities]]
        for city, row in zip(cities, matrix):
            matrix_rows.append([city.name] + [f"{value:.2f}" for value in row])
        story.append(_table(matrix_rows))

    story.append(_p("二、求解规划", styles["h2"]))
    story.append(_p(plan, styles["body"]))

    story.append(_p("三、模拟退火算法设置", styles["h2"]))
    params = solution["params"]
    param_rows = [
        ["参数", "取值", "含义"],
        ["初始温度 T0", f"{params['initial_temperature']}", "高温阶段提高搜索自由度"],
        ["降温系数 alpha", f"{params['cooling_rate']}", "指数降温 T = alpha * T"],
        ["马尔可夫链长度 L", f"{params['chain_length']}", "每个温度下的内循环次数"],
        ["终止温度 Tf", f"{params['final_temperature']}", "温度低于该值时停止"],
        ["随机种子", f"{params['seed']}", "保证实验可复现"],
    ]
    story.append(_table(param_rows, [4 * cm, 3.5 * cm, 8 * cm]))
    story.append(
        _p(
            "状态产生函数采用交换扰动：随机选取路线中的两个城市并交换位置。"
            "状态接受函数采用 Metropolis 准则：若新路线更短则直接接受；若新路线更长，则以 exp(-delta/T) 的概率接受。",
            styles["body"],
        )
    )

    story.append(_p("四、求解结果", styles["h2"]))
    result_rows = [
        ["指标", "结果"],
        ["城市数量", str(solution["city_count"])],
        ["总迭代次数", str(solution["total_iterations"])],
        ["最优路线", " -> ".join(solution["closed_route_names"])],
        ["最优总路程", f"{solution['best_length']:.4f}"],
        ["初始参考路程", f"{solution['initial_length']:.4f}"],
        ["改进比例", f"{solution['improvement_rate'] * 100:.2f}%"],
        ["接受坏解次数", str(solution["worse_accepted"])],
        ["最优解更新次数", str(solution["improved"])],
    ]
    story.append(_table(result_rows, [5 * cm, 10.5 * cm]))

    story.append(_p("五、代表性迭代过程", styles["h2"]))
    sample_rows = [["温度轮次", "内循环", "delta", "接受概率", "是否接受", "当前最优"]]
    for step in solution["sample_steps"][:8]:
        sample_rows.append(
            [
                str(step["temperature_step"]),
                str(step["inner_iteration"]),
                f"{step['delta']:.4f}",
                f"{step['acceptance_probability']:.4f}",
                "是" if step["accepted"] else "否",
                f"{step['best_length_after']:.4f}",
            ]
        )
    story.append(_table(sample_rows, [2.2 * cm, 2.2 * cm, 3 * cm, 3 * cm, 2.2 * cm, 3 * cm]))

    story.append(_p("六、可视化", styles["h2"]))
    story.append(Image(route_chart, width=15.5 * cm, height=11.3 * cm))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Image(convergence_chart, width=15.5 * cm, height=8.7 * cm))

    story.append(_p("七、结果解释", styles["h2"]))
    story.append(_p(summary, styles["body"]))
    story.append(
        _p(
            "结论：本次求解得到的闭合路线为 "
            f"{' -> '.join(solution['closed_route_names'])}，总路程为 {solution['best_length']:.4f}。"
            "从收敛过程看，模拟退火在高温阶段保持探索能力，并在降温过程中逐步稳定到较优路线。",
            styles["body"],
        )
    )

    doc.build(story)
    return str(path)
