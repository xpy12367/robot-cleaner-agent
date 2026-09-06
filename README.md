# 扫地机器人智能客服 Agent

一个基于 Streamlit、LangChain、LangGraph 和 Chroma 的中文扫地机器人客服 Agent。项目内置本地知识库，可用于选购咨询、故障初步排查、日常保养、天气适配和模拟月度使用报告生成。

## 主要功能

- ReAct Agent：根据用户问题选择知识库、天气、位置和模拟使用记录等工具。
- RAG 检索：从 `data/` 下的 TXT/PDF 知识文件中检索相关内容。
- 流式对话：通过 Streamlit 聊天界面逐步展示最终回答。
- 使用报告：根据项目内的模拟用户数据生成月度报告。
- 本地向量库：首次检索时根据知识文件生成 Chroma 索引，后续按文件变化增量更新。

## 项目结构

```text
.
├─ agent/              # Agent、工具与中间件
├─ config/             # 模型、提示词与向量库配置
├─ data/               # 知识库与演示数据
├─ model/              # 对话模型和 Embedding 工厂
├─ prompts/            # 系统、RAG 和报告提示词
├─ rag/                # 检索与向量库服务
├─ tests/              # 无网络最小启动测试
├─ utils/              # 配置、日志、路径和文件工具
├─ .streamlit/         # Streamlit 主题配置
├─ app.py              # Web 应用入口
└─ requirements.txt    # Python 依赖
```

## 环境要求

- Python 3.11 或更高版本（已在 Python 3.14.5 验证）
- 阿里云百炼 OpenAI 兼容模式的 API Key 和 Base URL
- 可选：高德 Web 服务 Key，用于 IP 定位和天气查询

## 安装

Windows PowerShell：

```powershell
git clone https://github.com/xpy12367/robot-cleaner-agent.git
cd robot-cleaner-agent

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 配置

`DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL` 是启动模型所必需的两项配置。Base URL 由所使用的服务、地域或专属实例决定，请以百炼控制台提供的地址为准。

### 方式一：PowerShell 环境变量

```powershell
$env:DASHSCOPE_API_KEY = "你的 API Key"
$env:DASHSCOPE_BASE_URL = "https://你的服务地址/compatible-mode/v1"

# 可选：启用定位与天气功能
$env:AMAP_KEY = "你的高德 Web 服务 Key"
```

### 方式二：Streamlit secrets

在本地新建 `.streamlit/secrets.toml`：

```toml
DASHSCOPE_API_KEY = "你的 API Key"
DASHSCOPE_BASE_URL = "https://你的服务地址/compatible-mode/v1"
```

`secrets.toml` 已被 `.gitignore` 忽略，请勿将真实凭据提交到 Git。也可启动页面后，在左侧边栏临时输入 API Key 和 Base URL；该方式只在当前 Streamlit 会话中生效。

`.env.example` 只用于展示字段，当前应用不会自动加载 `.env` 文件。

## 运行

```powershell
python -m streamlit run app.py
```

打开浏览器显示的本地地址即可使用。页面本身可在未配置密钥时启动；首次提问前必须配齐 API Key 和 Base URL。首次检索会创建本地 `chroma_db/` 索引，耗时取决于网络和知识库大小。

## 最小运行测试

不需要真实 API Key，也不会发起模型或天气网络请求：

```powershell
python -m unittest discover -s tests -v
```

测试会确认：

- Streamlit 应用可以完整渲染。
- API Key 和 Base URL 输入框存在。
- 不合法的 Base URL 会被界面拦截。

## 本地生成文件

以下内容不会提交到 Git：

- `.venv/`：本地 Python 虚拟环境
- `logs/`：运行日志
- `chroma_db/`：本地 Chroma 索引
- `md5.text`：知识文件变更状态
- `.idea/` 和 `__pycache__/`：IDE 与 Python 缓存

## 知识库来源与限制

知识库来源及适用边界见：

- [`data/external/来源清单.md`](data/external/来源清单.md)
- [`data/external/知识库元数据.md`](data/external/知识库元数据.md)

本项目用于通用问答和基础排障，不能替代具体型号说明书或官方售后意见。涉及拆机、锂电池、充电异常或其他安全风险时，应优先停止使用并联系厂商。

## 安全说明

- 不要将真实 API Key、个人 Base URL、`.env` 或 `secrets.toml` 提交到 Git。
- 项目中的用户位置、用户 ID 和使用记录为演示数据，不连接真实业务系统。
- 如真实凭据曾经进入本地文件或日志，建议在对应平台轮换该凭据。

## 许可证

当前仓库未附加开源许可证，默认保留全部权利。
