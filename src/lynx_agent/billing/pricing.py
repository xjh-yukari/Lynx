"""2026-10-24 核验的官方人民币单价，单位为元/百万 Tokens。"""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import urlsplit

VERSION = "2026-10-24"
SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
BEIJING = timezone(timedelta(hours=8))
# 按官方公开的放假日期估算；周末无论是否调休上班均按谷时处理。
# https://www.gov.cn/zhengce/content/202511/content_7047090.htm
HOLIDAYS_2026 = (("01-01", "01-03"), ("02-15", "02-23"), ("04-04", "04-06"),
                 ("05-01", "05-05"), ("06-19", "06-21"), ("09-25", "09-27"), ("10-01", "10-07"))
PEAK_CNY = {
    "deepseek-flash": (Decimal("0.04"), Decimal(2), Decimal(8)),
    "deepseek-v4-pro": (Decimal("0.30"), Decimal(9), Decimal(27)),
}
ALIASES = {"deepseek-v4-flash": "deepseek-flash", "deepseek-v4-flash-vision-exp": "deepseek-flash"}


def utc_now():
    return datetime.now(UTC)


def official_endpoint(url):
    parsed = urlsplit(str(url))
    return (parsed.scheme == "https" and parsed.hostname == "api.deepseek.com"
            and parsed.port in {None, 443} and parsed.path.rstrip("/") in {"", "/v1", "/beta"})


def period_at(instant):
    """北京时间半开区间；未核验年份的候选峰时不猜节假日。"""
    if instant.tzinfo is None:
        raise ValueError("计费时段需要带时区的时间")
    local = instant.astimezone(BEIJING)
    minute = local.hour * 60 + local.minute
    if local.weekday() >= 5 or not (540 <= minute < 720 or 840 <= minute < 1080):
        return "off_peak"
    if local.year != 2026:
        return None
    day = local.strftime("%m-%d")
    return "off_peak" if any(start <= day <= end for start, end in HOLIDAYS_2026) else "peak"


def period_label(instant=None):
    return {"peak": "峰", "off_peak": "谷", None: "时段待核验"}[period_at(instant or utc_now())]


def rates_for(model, started_at, official=True):
    rates = PEAK_CNY.get(ALIASES.get(model, model))
    period = period_at(started_at)
    if not official or rates is None or period is None:
        return None
    factor = Decimal("0.5") if period == "off_peak" else Decimal(1)
    return tuple(rate * factor for rate in rates)


def price_components(model, usage, started_at, official=True):
    """缺缓存拆分/总用量/模型价格/日历时，不把未知费用计成零。"""
    rates = rates_for(model, started_at, official)
    if rates is None or usage is None:
        return None
    cached = usage.cached_prompt_tokens
    if cached is None and usage.prompt_tokens:
        return None
    cached = cached or 0
    hit, miss, output = rates
    return (cached * hit / Decimal(1000000),
            (usage.prompt_tokens - cached) * miss / Decimal(1000000),
            usage.completion_tokens * output / Decimal(1000000))


def price_usage(model, usage, started_at, official=True):
    parts = price_components(model, usage, started_at, official)
    return sum(parts, Decimal(0)) if parts is not None else None
