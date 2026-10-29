"""
扫地机器人智能客服 —— Streamlit Web 前端页面
==============================================
运行方式(在项目根目录下执行):
    streamlit run app.py

页面功能:
    1. 聊天式交互:点击推荐问题或手动输入提问,客服回答以打字机效果流式输出
    2. 只展示最终回答:自动过滤模型的思考过程与工具中间输出(见 react_agent.execute_chat_stream)
    3. 侧边栏:模型连接配置 / 快捷提问 / 清空对话 / 系统信息
"""
import os
import sys
from itertools import chain
from urllib.parse import urlparse
import yaml
import streamlit as st
from dotenv import load_dotenv

# 将项目根目录加入模块搜索路径,确保页面内可以导入 agent / rag / utils 等包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
# override=False ：如果系统环境变量里已经有值，就优先使用系统环境变量；没有时才读取 .env
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="扫地机器人智能客服",
    page_icon="🤖",
    layout="centered",
)

# ==================== 常量 ====================
PAGE_TITLE = "扫地机器人智能客服"
AGENT_AVATAR = ":material/smart_toy:"   # 客服头像(材料符号图标)
USER_AVATAR = ":material/person:"       # 用户头像

# 推荐问题:key=展示标签  value=点击后发送给智能体的完整问题
RECOMMENDED_QUESTIONS = {
    "选购推荐": "小户型适合买哪种扫地机器人?",
    "故障排查": "扫地机器人常见的故障有哪些?分别怎么排查解决?",
    "日常保养": "扫地机器人日常应该如何保养维护?",
    "天气适配": "我这边近期潮湿多雨,扫地机器人还能正常使用吗?",
    "宠物家庭": "家里养了宠物,应该怎么选扫拖一体机器人?",
    "使用报告": "帮我生成我的扫地机器人月度使用报告",
}

WELCOME_MESSAGE = (
    "您好!我是扫地机器人智能客服,可以为您解答 **选购、使用、故障排查、维护保养** 等问题,"
    "也可以调用工具为您生成 **个人使用报告**。请问有什么可以帮您?"
)


def load_agent_config() -> dict:
    """读取 agent.yaml 中的模型与环境配置(轻量读取,避免触发后端重模块的导入)"""
    config_path = os.path.join(PROJECT_ROOT, "config", "agent.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="正在初始化智能客服并检查知识库…")
def get_agent():
    """
    懒加载并缓存智能体实例:第一次提问时才导入后端模块并完成初始化

    注意: 使用Streamlit的session_id作为Agent的会话ID,确保每个用户有独立的记忆
    """
    from agent.react_agent import ReactAgent

    # 获取Streamlit的session_id作为唯一标识
    # 如果没有session_id（旧版本Streamlit），使用默认值
    session_id = getattr(st.session_state, "_session_id", "default")

    return ReactAgent(
        session_id=session_id,
        enable_memory=True,  # 启用会话记忆
        max_history_turns=5   # 保留最近5轮对话
    )


@st.cache_data(ttl="10m", max_entries=1)
def knowledge_files() -> list[str]:
    """列出 data 目录下的知识库文件(仅用于侧边栏信息展示)"""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    return [
        name for name in sorted(os.listdir(data_dir))
        if name.lower().endswith((".txt", ".pdf"))
    ]


def get_env_name(config_key: str, default: str) -> str:
    """获取 YAML 中指定的环境变量名。"""
    return load_agent_config().get(config_key, default)


def get_setting_source(config_key: str, session_key: str, default_env: str) -> str:
    """检测连接配置来自环境变量、secrets.toml 或当前会话。"""
    env_name = get_env_name(config_key, default_env)
    if os.getenv(env_name):
        return "环境变量"
    if st.session_state.get(session_key):
        return "手动配置(本次会话)"
    try:
        if st.secrets.get(env_name):
            return "secrets.toml"
    except Exception:
        pass
    return ""


def ensure_connection_settings() -> list[str]:
    """
    在模型工厂导入前，把 API Key 和 Base URL 注入各自的环境变量。
    返回仍未配置的项目名称。
    """
    settings = (
        ("API Key", "api_key", "api_key", "DASHSCOPE_API_KEY"),
        ("Base URL", "base_url", "base_url", "DASHSCOPE_BASE_URL"),
    )
    missing = []

    for label, config_key, session_key, default_env in settings:
        env_name = get_env_name(config_key, default_env)
        value = os.getenv(env_name) or st.session_state.get(session_key)
        if not value:
            try:
                value = st.secrets.get(env_name)
            except Exception:
                value = None
        if value:
            if config_key == "base_url" and not is_valid_base_url(str(value)):
                missing.append("Base URL（格式无效）")
                continue
            os.environ[env_name] = str(value).strip()
            st.session_state[session_key] = str(value).strip()
        else:
            missing.append(label)

    return missing


def is_valid_base_url(value: str) -> bool:
    """只接受带主机名的 HTTP(S) Base URL。"""
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def save_connection_settings(api_key: str, base_url: str) -> str:
    """保存当前会话提交的连接配置，成功时返回空字符串。"""
    api_key = api_key.strip()
    base_url = base_url.strip().rstrip("/")

    if not api_key and not base_url:
        return "请至少填写一项需要更新的连接配置"
    if base_url and not is_valid_base_url(base_url):
        return "Base URL 必须是以 http:// 或 https:// 开头的有效地址"

    if api_key:
        st.session_state["api_key"] = api_key
        os.environ[get_env_name("api_key", "DASHSCOPE_API_KEY")] = api_key
    if base_url:
        st.session_state["base_url"] = base_url
        os.environ[get_env_name("base_url", "DASHSCOPE_BASE_URL")] = base_url

    return ""


def handle_prompt(question: str) -> None:
    """处理一次用户提问:回显问题,并以打字机效果流式渲染客服回答"""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    missing_settings = ensure_connection_settings()
    if missing_settings:
        missing_text = "、".join(missing_settings)
        st.error(f"未检测到可用的 {missing_text},请先在左侧边栏完成配置后再提问")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"> ⚠️ 服务不可用:未配置 {missing_text}"}
        )
        return

    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        try:
            # 模型思考与工具调用阶段可能持续数秒,先用等待提示占位
            with st.spinner("正在思考,检索知识库 / 调用工具…"):
                stream = get_agent().execute_chat_stream(question)
                # 阻塞至第一个文本 token,即思考阶段结束(后续交给 write_stream 流式渲染)
                first_chunk = next(stream, None)
            st.session_state["agent_ready"] = True
        except Exception as exc:
            st.error(f"智能客服调用失败:{type(exc).__name__} {exc}")
            st.caption("详细错误日志请查看项目 logs 目录下的日志文件")
            st.session_state.messages.append(
                {"role": "assistant", "content": "> ⚠️ 本次服务调用失败,请稍后重试"}
            )
            return

        if not first_chunk:
            st.warning("本次没有获取到有效回答,换个问法再试试吧")
            st.session_state.messages.append(
                {"role": "assistant", "content": "(未获取到有效回答)"}
            )
            return

        try:
            # chain 把预取的首段文本接回剩余流,实现完整的打字机效果
            full_answer = st.write_stream(chain([first_chunk], stream))
        except Exception as exc:
            st.error(f"回答生成中断:{type(exc).__name__} {exc}")
            full_answer = "(回答生成中断,可重新提问)"
        st.session_state.messages.append({"role": "assistant", "content": full_answer})


@st.dialog("清空对话")
def confirm_clear():
    """弹窗确认后清空当前会话记录"""
    st.write("确定要清空当前会话记录吗?智能客服将重新开始对话。")
    left, right = st.columns(2)
    if left.button("清空", type="primary", use_container_width=True):
        # 清空Streamlit前端的消息历史
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]

        # 清空Agent的会话记忆
        try:
            agent = get_agent()
            agent.clear_memory()
        except Exception as e:
            st.error(f"清空Agent记忆失败: {e}")

        st.rerun()
    if right.button("取消", use_container_width=True):
        st.rerun()


# ==================== 会话状态初始化 ====================
st.session_state.setdefault(
    "messages", [{"role": "assistant", "content": WELCOME_MESSAGE}]
)
st.session_state.setdefault("agent_ready", False)
st.session_state.setdefault("pending_question", None)

# ==================== 页面顶部(标题区) ====================
st.html(
    """
    <style>
        .agent-hero { text-align: center; padding: 0.8rem 0 0.2rem 0; }
        .agent-hero .agent-logo { font-size: 2.6rem; line-height: 1.3; }
        .agent-hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0.1rem 0 0.3rem 0; }
        .agent-hero p { opacity: 0.7; margin: 0; }
    </style>
    <div class="agent-hero">
        <div class="agent-logo">🤖</div>
        <h1>扫地机器人智能客服</h1>
        <p>ReAct 智能体 · 知识库检索 · 天气适配 · 使用报告生成</p>
    </div>
    """
)
st.divider()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 服务状态")
    if st.session_state.get("agent_ready"):
        st.markdown(":green[● 服务运行中]")
    else:
        st.markdown(":orange[● 等待首次提问]")
        st.caption("首次提问时会检查知识库；首次构建可能需要数分钟")

    st.markdown("---")

    # —— 模型连接配置 ——
    key_source = get_setting_source(
        "api_key", "api_key", "DASHSCOPE_API_KEY"
    )
    base_url_source = get_setting_source(
        "base_url", "base_url", "DASHSCOPE_BASE_URL"
    )
    with st.container(border=True):
        st.markdown("**模型连接**")
        key_status = f":green[已配置 · {key_source}]" if key_source else ":red[未配置]"
        base_url_status = (
            f":green[已配置 · {base_url_source}]" if base_url_source else ":red[未配置]"
        )
        st.markdown(f"- API Key：{key_status}\n- Base URL：{base_url_status}")

        with st.form("connection_settings", border=False):
            new_key = st.text_input(
                "API Key",
                type="password",
                placeholder="留空表示不修改",
                key="api_key_input",
            )
            new_base_url = st.text_input(
                "Base URL",
                placeholder="https://你的服务地址/v1",
                key="base_url_input",
            )
            submitted = st.form_submit_button(
                "保存连接配置",
                icon=":material/settings_ethernet:",
                width="stretch",
            )

        if submitted:
            error_message = save_connection_settings(new_key, new_base_url)
            if error_message:
                st.error(error_message)
            else:
                get_agent.clear()
                st.session_state["agent_ready"] = False
                st.rerun()

        st.caption("也可使用 DASHSCOPE_API_KEY 与 DASHSCOPE_BASE_URL 环境变量")

    st.markdown("---")

    # —— 快捷提问 ——
    st.markdown("**快捷提问**")
    for label, question in RECOMMENDED_QUESTIONS.items():
        if st.button(label, key=f"faq_{label}", width="stretch"):
            st.session_state["pending_question"] = question

    st.markdown("---")

    # —— 会话管理 ——
    if st.button("清空对话", icon=":material/delete_sweep:", width="stretch"):
        confirm_clear()

    st.markdown("---")

    # —— 系统信息 ——
    with st.expander("系统信息", expanded=False):
        agent_cfg = load_agent_config()
        st.markdown(f"- 对话模型:**{agent_cfg.get('chat_model_name', '-')}**")
        st.markdown(f"- 向量模型:**{agent_cfg.get('embedding_model_name', '-')}**")
        st.markdown(f"- 知识文件:**{len(knowledge_files())} 份**")

        # 显示会话记忆状态
        if st.session_state.get("agent_ready"):
            try:
                agent = get_agent()
                if agent.enable_memory:
                    turn_count = agent.memory.get_turn_count() if agent.memory else 0
                    st.markdown(f"- 记忆轮次:**{turn_count}/{agent.max_history_turns} 轮**")
                    st.caption("超过最大轮次后会自动清理最早的对话")
                else:
                    st.markdown("- 会话记忆:**未启用**")
            except Exception:
                pass

        st.caption("知识库来源:data 目录下的官方知识文档与外部使用记录")

    st.caption("智能客服前端 v1.0 · Powered by Streamlit")

# ==================== 聊天区 ====================
# 渲染历史消息
for message in st.session_state.messages:
    avatar = AGENT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 空对话时的推荐问题(已有提问或正在处理提问时不展示)
has_user_message = any(m.get("role") == "user" for m in st.session_state.messages)
suggested_question = None
if not has_user_message and not st.session_state.get("pending_question"):
    st.caption("可以试试这样问 👇")
    picked = st.pills(
        "推荐问题",
        options=list(RECOMMENDED_QUESTIONS.keys()),
        label_visibility="collapsed",
    )
    if picked:
        suggested_question = RECOMMENDED_QUESTIONS[picked]

# 提问入口
prompt = st.chat_input("请输入您的问题,例如:帮我生成我的扫地机器人使用报告")

# 统一处理提问来源:侧边栏快捷提问 > 输入框 > 推荐问题
question = st.session_state.pop("pending_question", None)
if not question and prompt:
    question = prompt
if not question and suggested_question:
    question = suggested_question

if question and question.strip():
    handle_prompt(question)
