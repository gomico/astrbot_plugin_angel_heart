"""判定人格摘要存储与生成。

秘书判定的 system 提示词默认使用会话人格全文；``analyzer_persona_mode=digest``
时改用每张人格卡（AstrBot v3 的 persona_id 即 name）单独生成的「判定用摘要」。

摘要只服务四件事：要不要接话、对谁说、用什么策略、要点是什么；不教说话风格，
也不含工具/生图/搜索/技能/记忆段落。按人格卡落盘到插件数据目录
``analyzer_personas.json``，不缓存人格原文、不做 hash 自动失效、不做定时重生成。

生成失败只记 warning，不落条目、不覆盖已有内容；判定链路始终 fail-open：
摘要缺失时回落人格全文，不等待生成。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from astrbot.api import logger

STORE_FILE_NAME = "analyzer_personas.json"

# 摘要长度校验：太短说明模型没按要求输出，太长说明把「怎么说」又写回来了
DIGEST_MIN_CHARS = 80
DIGEST_MAX_CHARS = 600

# 串行生成，每张人格卡之间留一点间隔，避免连续打满秘书模型
GENERATE_INTERVAL_SECONDS = 0.5

SOURCE_AUTO = "auto"
SOURCE_MANUAL = "manual"

# 抽取指令：原样固定，不要改写（改它等于改判定输入）
DIGEST_INSTRUCTION = """你在为一套「要不要在群聊里发言」的判定器准备人格摘要。
下面给出的是一张人格卡的全文。判定器只负责四件事：这句话要不要接、该对谁说、用什么策略说、要点是什么。
它不负责把话说得好看。

因此摘要里只保留会影响「判断」的信息，所有关于「怎么说」的内容全部丢掉。
人格卡里关于工具、生图、搜索、技能、记忆系统的段落与本摘要无关，直接忽略。

按下面四行输出，每行以标签开头。不要编号、不要 Markdown、不要 JSON、不要任何额外解释：

身份：一句话说明是谁、在什么场合说话。
会接的话题：列出人格卡里提到的兴趣、爱好、知识领域、擅长的事，用顿号分隔，要具体（写「音游、射击游戏」，不写「游戏」）。
雷区：列出不参与的话题、必须闭嘴的情形、对什么人或事有避讳。
长度上限：一次最多几个字、几句。

人格卡里没写的项，那一行就写「未写明」，不要推测、不要补全。
四行合计不超过 400 字。"""


def _now_iso() -> str:
    """当前时间（带本地时区偏移），与数据模型里的 generated_at 一致。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class DigestStore:
    """人格卡判定摘要的 JSON 持久化存储（全局一份，key 为人格 name）。"""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._file_path = os.path.join(data_dir, STORE_FILE_NAME)
        self._lock = threading.Lock()
        self._personas: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    # ---------- 持久化 ----------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data: Dict[str, Any] = {}
        if os.path.isfile(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError, TypeError):
                data = {}
        if not isinstance(data, dict):
            data = {}
        personas = data.get("personas", {})
        if isinstance(personas, dict):
            self._personas = {
                str(name): dict(entry)
                for name, entry in personas.items()
                if name and isinstance(entry, dict)
            }
        self._loaded = True

    def _save_locked(self) -> None:
        """写盘（调用方必须已持锁）。"""
        payload = {"version": 1, "personas": self._personas}
        tmp_path = self._file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._file_path)

    # ---------- 读写 ----------

    def get(self, name: Any) -> str:
        """取某人格局限的摘要；条目缺失或摘要为空串返回空字符串。"""
        key = str(name or "").strip()
        if not key:
            return ""
        with self._lock:
            self._ensure_loaded()
            entry = self._personas.get(key)
            if not isinstance(entry, dict):
                return ""
            return str(entry.get("digest") or "").strip()

    def entries(self) -> Dict[str, Dict[str, Any]]:
        """全部条目快照（深拷贝），供 WebUI 读取。"""
        with self._lock:
            self._ensure_loaded()
            return {
                name: dict(entry) for name, entry in self._personas.items()
            }

    def set(
        self, name: Any, digest: Any, source: Any, model: Any = ""
    ) -> Optional[Dict[str, Any]]:
        """写入一条摘要；name 为空时返回 None。

        ``source`` 只接受 auto/manual，其余按 manual 处理；``generated_at``
        一律取当前时间。空的 digest 也落盘（等价于待生成），由 missing() 识别。
        """
        key = str(name or "").strip()
        if not key:
            return None
        text = str(digest or "").strip()
        normalized_source = SOURCE_AUTO if source == SOURCE_AUTO else SOURCE_MANUAL
        entry = {
            "digest": text,
            "generated_at": _now_iso(),
            "model": str(model or ""),
            "source": normalized_source,
        }
        with self._lock:
            self._ensure_loaded()
            self._personas[key] = entry
            self._save_locked()
            return dict(entry)

    def reset(self, name: Any) -> bool:
        """删除某张人格卡的摘要条目；返回是否真的删掉了。"""
        key = str(name or "").strip()
        if not key:
            return False
        with self._lock:
            self._ensure_loaded()
            if key not in self._personas:
                return False
            del self._personas[key]
            self._save_locked()
            return True

    def missing(self, names: Iterable[Any]) -> List[str]:
        """从给定人格卡里挑出「待生成」（条目缺失或摘要为空串）的 name 列表。"""
        result: List[str] = []
        seen: set[str] = set()
        for raw in names or []:
            key = str(raw or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            if not self.get(key):
                result.append(key)
        return result


async def generate(
    provider: Any, persona_text: str, model_name: str
) -> Optional[str]:
    """用秘书模型把一张人格卡压成判定摘要；失败返回 None（只记 warning）。"""
    if provider is None:
        logger.warning(
            f"AngelHeart: 判定人格摘要生成失败，未找到秘书模型提供商 '{model_name}'"
        )
        return None

    text = str(persona_text or "").strip()
    if not text:
        logger.warning("AngelHeart: 判定人格摘要生成跳过，人格卡正文为空")
        return None

    try:
        response = await provider.text_chat(
            prompt=DIGEST_INSTRUCTION,
            contexts=[{"role": "user", "content": text}],
            system_prompt=None,
        )
        output = str(getattr(response, "completion_text", "") or "").strip()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"AngelHeart: 判定人格摘要生成调用失败（{model_name}）: {e}")
        return None

    length = len(output)
    if not (DIGEST_MIN_CHARS <= length <= DIGEST_MAX_CHARS):
        logger.warning(
            f"AngelHeart: 判定人格摘要长度不合规（{length} 字符，要求 "
            f"{DIGEST_MIN_CHARS}-{DIGEST_MAX_CHARS}），已丢弃: "
            f"{output[:80]!r}"
        )
        return None
    return output


async def ensure_all(
    config_manager: Any,
    provider_getter: Callable[[str], Any],
    persona_resolver: Callable[[str], Any],
    umos: Sequence[str],
    store: DigestStore,
) -> None:
    """串行补生成缺失的人格摘要（后台任务调用，不阻塞判定链路）。

    Args:
        config_manager: 读取 analyzer_model。
        provider_getter: 模型名 -> provider（找不到返回 None）。
        persona_resolver: 会话 umo -> (人格 name, 人格正文)，必须与判定链路
            使用同一条解析链（``LLMAnalyzer.resolve_persona``）。
        umos: 本插件服务的会话列表。
        store: 摘要存储。
    """
    if store is None:
        return

    model_name = ""
    try:
        model_name = str(config_manager.analyzer_model or "").strip() if config_manager else ""
    except Exception:
        model_name = ""
    if not model_name:
        logger.warning(
            "AngelHeart: 秘书模型未配置，跳过判定人格摘要生成"
        )
        return

    # 先枚举出「缺失摘要的人格卡」，同一张卡只排一次
    seen: set[str] = set()
    cards: List[tuple[str, str]] = []
    for umo in umos or []:
        try:
            name, prompt = await persona_resolver(str(umo))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                f"AngelHeart: 解析会话人格失败，跳过该会话的摘要生成（{umo}）: {e}"
            )
            continue
        card_name = str(name or "").strip()
        card_text = str(prompt or "").strip()
        if not card_name or not card_text or card_name in seen:
            continue
        seen.add(card_name)
        if store.get(card_name):
            continue
        cards.append((card_name, card_text))

    if not cards:
        return

    provider = None
    try:
        provider = provider_getter(model_name) if provider_getter else None
    except Exception as e:
        logger.warning(f"AngelHeart: 获取秘书模型提供商失败: {e}")
        provider = None
    # 提供商缺失属于配置错误：只记一条，避免每张卡刷一遍 warning
    if provider is None:
        logger.warning(
            f"AngelHeart: 秘书模型提供商 '{model_name}' 不可用，"
            f"跳过 {len(cards)} 张人格卡的摘要生成"
        )
        return

    for index, (card_name, card_text) in enumerate(cards):
        if index:
            await asyncio.sleep(GENERATE_INTERVAL_SECONDS)
        digest = await generate(provider, card_text, model_name)
        if not digest:
            continue
        store.set(card_name, digest, SOURCE_AUTO, model_name)
        logger.info(
            f"AngelHeart: 已生成判定人格摘要「{card_name}」（{len(digest)} 字符）"
        )


def collect_service_umos(
    config_manager: Any,
    profile_store: Any,
    chat_sources: Any = None,
) -> List[str]:
    """枚举本插件服务的会话：access_control.chat_ids ∪ 已绑定配置模板的会话。

    白名单只存纯群号，尽量用来源登记展开成完整 umo（session_service_config
    这类按 umo 作用域的配置只有完整 origin 才读得到）。找不到对应来源时保留
    纯群号，让上层按现有降级路径处理。
    """
    candidates: List[str] = []

    try:
        bindings = profile_store.list_bindings() if profile_store is not None else []
    except Exception:
        logger.debug("AngelHeart: 读取群聊绑定失败", exc_info=True)
        bindings = []
    for item in bindings or []:
        if isinstance(item, dict):
            candidates.append(str(item.get("chat_id") or "").strip())

    try:
        whitelist = list(config_manager.chat_ids or []) if config_manager else []
    except Exception:
        logger.debug("AngelHeart: 读取白名单群聊失败", exc_info=True)
        whitelist = []
    for raw in whitelist:
        candidates.append(str(raw or "").strip())

    sources: List[Dict[str, Any]] = []
    if chat_sources is not None:
        try:
            sources = chat_sources.list_sources() or []
        except Exception:
            logger.debug("AngelHeart: 读取来源登记失败", exc_info=True)
            sources = []

    umos: List[str] = []
    expanded: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        umo = _expand_to_origin(candidate, sources)
        if umo:
            expanded.append(umo)

    # 先保留完整 origin，再补没有来源登记的纯群号；同号（纯群号/完整 origin）
    # 只留一条，避免同一会话被解析两次
    seen: set[str] = set()
    for umo in expanded:
        if ":" not in umo:
            continue
        key = umo.split(":")[-1]
        if key in seen:
            continue
        seen.add(key)
        umos.append(umo)
    for umo in expanded:
        if ":" in umo or umo in seen:
            continue
        seen.add(umo)
        umos.append(umo)
    return umos


def _expand_to_origin(candidate: str, sources: Sequence[Dict[str, Any]]) -> str:
    """纯群号尽量展开成来源登记里的完整 umo；已是完整 origin 时原样返回。"""
    if ":" in candidate:
        return candidate
    for entry in sources or []:
        if not isinstance(entry, dict):
            continue
        chat_id = str(entry.get("chat_id") or "").strip()
        if not chat_id or ":" not in chat_id:
            continue
        # 纯群号只在群聊来源里匹配，避免同号私聊串台
        if str(entry.get("kind") or "") == "private":
            continue
        if chat_id.split(":")[-1] == candidate:
            return chat_id
    return candidate
