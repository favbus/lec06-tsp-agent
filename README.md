# LEC06 TSP 优化智能体

这是一个基于公开大模型 API、LangGraph 和模拟退火算法的旅行商问题 TSP 优化智能体。

## 功能

- 支持随机生成城市或粘贴城市坐标。
- 使用 LangGraph 串联智能体流程：问题解析、求解规划、工具调用、结果总结、PDF 生成。
- 使用模拟退火算法求解 TSP，默认参数参考第九章模拟退火课件。
- 自动输出 PDF 求解报告，可用于作业提交。

## 运行

```bash
conda activate lec06-tsp-agent
streamlit run app.py
```

如果不想激活环境，也可以：

```bash
conda run -n lec06-tsp-agent streamlit run app.py
```

## API 配置

项目读取 `.env` 中的配置：

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.asxs.top/v1
OPENAI_MODEL=gpt-5.5
```

如果 API 临时不可用，系统会自动降级为本地规则说明，模拟退火求解和 PDF 输出仍然可用。

## Streamlit Cloud 部署

入口文件：

```text
app.py
```

Secrets 配置：

```toml
OPENAI_API_KEY = "你的 key"
OPENAI_BASE_URL = "https://api.asxs.top/v1"
OPENAI_MODEL = "gpt-5.5"
```

## 推荐录屏流程

1. 打开 Streamlit 页面。
2. 选择“课件 4 城市示例”或随机生成 12 个城市。
3. 点击“运行智能体求解”。
4. 展示智能体流程、最优路径、收敛曲线和 PDF 下载。
5. 下载 PDF 作为提交材料之一。
