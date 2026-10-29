import json
import os
import csv
import random
from datetime import datetime
import urllib.request   # Python 内置的发 HTTP 请求的库
from urllib.parse import urlencode
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

rag=RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]

external_data = {}


# ==================== 高德 定位 + 天气 ====================
def _amap_key() -> str:
    """读取高德Web服务Key（agent.yaml 的 amap_key 存的是环境变量名）"""
    return os.getenv(agent_config.get("amap_key", "AMAP_KEY"))
    #.get("amap_key", "AMAP_KEY")：取不到就用默认值 "AMAP_KEY"


def _http_get_json(url: str, timeout: int = 5) -> dict:
    """发起GET请求并解析JSON；失败返回空dict，避免工具崩溃"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp: #urlopen(url) 发 GET 请求。 urllib.request:Python 内置的发 HTTP 请求的库
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"[_http_get_json]请求失败: {url.split('?')[0]}, 原因: {e}")
        return {}


def get_user_city_by_ip() -> str:
    """
    通过高德IP定位获取本机出口IP所在城市。
    注意:不带ip参数时按服务器出口IP定位——本机/localhost演示时即等于"当前用户所在地"。
    :return: 标准城市名(如"深圳")；失败返回空串
    """
    key = _amap_key()
    if not key:
        logger.warning("[get_user_city_by_ip]未配置高德Key(环境变量AMAP_KEY)")
        return ""
    #urlencode({"key": key}) 会把 {"key": "xxxx"} 拼成 key=xxxx 作为网址参数
    data = _http_get_json("https://restapi.amap.com/v3/ip?" + urlencode({"key": key}))
    if not data or data.get("status") != "1": #data.get("status") != "1"：高德接口成功返回 status:"1"，其它都是失败。
        logger.warning(f"[get_user_city_by_ip]定位失败: {data.get('info', '无响应') if data else '无响应'}")
        return ""
    city = (data.get("city") or "").strip()
    if not city or city in ("[]", "[] "):  # 直辖市(北京/上海/天津/重庆)city为空，回退到省名
        city = data.get("province", "").strip()
    return city.replace("市", "").strip() #city.replace("市", "")：把 "深圳市" 去掉"市"，统一成 "深圳"，方便后面当参数传。


# 天气接口要的不是城市名，而是行政区划代码 adcode（一串数字，如深圳是 440300）
def _city_to_adcode(city: str, key: str) -> str:
    """城市名 -> 高德adcode（天气接口入参）"""
    data = _http_get_json(
        "https://restapi.amap.com/v3/geocode/geo?" + urlencode({"key": key, "address": city})
    )
    geocodes = data.get("geocodes") or [] # 取不到就当成空列表,避免报错
    return geocodes[0].get("adcode", "") if geocodes else ""  #.get("adcode", "")：取不到就用默认值 ""。


def fetch_city_weather(city: str) -> str:
    """按城市名查询高德实时天气，返回中文描述串；失败返回空串"""
    key = _amap_key()
    if not key:
        logger.warning("[fetch_city_weather]未配置高德Key(环境变量AMAP_KEY)")
        return ""
    adcode = _city_to_adcode(city, key)
    if not adcode:
        logger.warning(f"[fetch_city_weather]无法解析城市 {city} 的adcode")
        return ""
    data = _http_get_json(
        "https://restapi.amap.com/v3/weather/weatherInfo?"  #调高德实时天气接口
        + urlencode({"key": key, "city": adcode})
    )
    lives = data.get("lives") or []
    if not lives:
        logger.warning(f"[fetch_city_weather]未查询到城市 {city} 的天气数据")
        return ""
    l = lives[0]
    wind_dir = l.get("winddirection", "")
    wind_power = l.get("windpower", "").replace("级", "")
    wind_txt = f"{wind_dir}风{wind_power}级" if wind_power else (f"{wind_dir}风" if wind_dir else "微风")
    return (
        f"城市{l.get('city')}天气为{l.get('weather')}，气温{l.get('temperature')}摄氏度，"
        f"空气湿度{l.get('humidity')}%，{wind_txt}，天气观测时间{l.get('reporttime')}"
    )


@tool(description="从向量存储中检索参考资料")  # description: 描述工具的功能
def rag_summarize(query: str) -> str:
    return rag.Rag_summarize(query)

@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    try:
        result = fetch_city_weather(city)
        if result:
            return result
    except Exception as e:
        logger.error(f"[get_weather]查询 {city} 天气异常: {e}")
    return f"抱歉，暂时无法获取城市{city}的实时天气，请稍后再试。"

@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    try:
        city = get_user_city_by_ip()
        if city:
            return city
    except Exception as e:
        logger.error(f"[get_user_location]IP定位异常: {e}")
    logger.warning("[get_user_location]IP定位失败，回退到示例城市")
    return random.choice(["深圳", "合肥", "杭州"])

@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)

@tool(description="获取当前月份，以纯字符申形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def generate_external_data():
    """
    读取演示用户使用记录。
    原始日期如 2026-06-01，会统一转换为月份 2026-06。
    """
    if external_data:
        return

    external_data_path = get_abs_path(agent_config["external_data_path"])

    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

    with open(external_data_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f): # 读取CSV文件，返回一个字典列表
            user_id = (row.get("用户ID") or "").strip()
            date_text = (row.get("时间") or "").strip()

            if not user_id or len(date_text) < 7:
                logger.warning(f"[generate_external_data]跳过无效记录: {row}")
                continue

            month = date_text[:7]

            external_data.setdefault(user_id, {})[month] = {
                "特征": (row.get("特征") or "").strip(),
                "效率": (row.get("清洁效率") or "").strip(),
                "耗材": (row.get("耗材") or "").strip(),
                "对比": (row.get("对比") or "").strip(),
            }

    logger.info(f"[generate_external_data]已加载 {len(external_data)} 个演示用户")


@tool(
    description=(
        "从外部系统获取指定用户在指定月份的使用记录。"
        "返回 JSON 字符串；如果指定月份不存在，会返回最近可用月份，"
        "并通过 used_latest_available 字段明确标记。"
    )
)
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    user_records = external_data.get(user_id)
    if not user_records:
        logger.warning(f"[fetch_external_data]未找到用户 {user_id} 的使用记录")
        return ""

    requested_month = month.strip()[:7]

    if requested_month in user_records:
        data_month = requested_month
        used_latest_available = False
    else:
        data_month = max(user_records)
        used_latest_available = True
        logger.warning(
            f"[fetch_external_data]用户 {user_id} 没有 {requested_month} 的记录，"
            f"回退到最近可用月份 {data_month}"
        )

    payload = {
        "requested_month": requested_month,
        "data_month": data_month,
        "used_latest_available": used_latest_available,
        "record": user_records[data_month],
    }

    return json.dumps(payload, ensure_ascii=False) # 转换为 JSON 字符串，确保中文字符正常显示


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

if __name__ == '__main__':
    print(fetch_external_data("1001", "2026-06-01"))
