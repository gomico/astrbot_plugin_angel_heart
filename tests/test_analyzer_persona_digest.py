"""判定人格摘要（analyzer_persona_mode）测试。

覆盖：
- full / digest 两种模式抓到的秘书判定请求体（system 与两条 user 消息）；
- 摘要生成（长度校验、抽取指令、provider 缺失/失败回落）；
- ensure_all 串行补生成、同卡去重、失败不落盘；
- WebUI 三条接口（列表/保存/清空）；
- 会话枚举与配置回落。

conftest.py 已建好 astrbot 桩；logger 是 MagicMock，用 call_args_list 断言日志次数。
"""

from __future__ import annotations

import json
import types

import pytest

from astrbot.api import logger as astrbot_logger

from astrbot_plugin_angel_heart.core.config_manager import ConfigManager
from astrbot_plugin_angel_heart.core.llm_analyzer import LLMAnalyzer
from astrbot_plugin_angel_heart.core.persona_digest import (
    DIGEST_INSTRUCTION,
    DIGEST_MAX_CHARS,
    DigestStore,
    collect_service_umos,
    ensure_all,
    generate,
)

CHAT_ID = "default:GroupMessage:10001"
PERSONA_NAME = "伊莉雅"
OTHER_PERSONA = "美游"

# 与线上群聊人格同量级（约 5289 字符）的假人格卡：正文里刻意保留工具/生图/搜索/说话风格段，
# 用来验证 digest 模式确实只吃摘要，而不是把全文塞进 system。
PERSONA_TARGET_LEN = 5289
PERSONA_SECTIONS = [
    "你是伊莉雅，冬木市的魔法少女，偶尔在游戏群里陪人聊天。",
    "",
    "## 身份",
    "伊莉雅·冯·爱因兹贝伦，外表是小女孩，实际年纪比看起来大，嘴硬心软。",
    "",
    "## 说话风格",
    "短句为主，一句话里有起承转合，不用书面语，也不端着。",
    "遇到自己感兴趣的话题会连着说两三句，遇到不感兴趣的只回一个「哦」。",
    "被人点名时会先反驳一句，再回答问题；被夸的时候会假装不在意。",
    "不喜欢说教，也不会主动安慰人，最多损两句。",
    "",
    "## 说话示例",
    "- 群友：今晚还打吗？ 伊莉雅：打啊，等我写完作业……不是，等我吃完饭。",
    "- 群友：这个boss好难。 伊莉雅：难的是你，不是我。",
    "",
    "## 工具分工",
    "angel_describe_image 用来看图，只在需要确认图片细节时调用。",
    "不要为了凑字数反复描述同一张图。",
    "",
    "## 生图",
    "生图前先确认画风与构图，默认二次元立绘，不要真人风。",
    "一次只出一张，出完先问要不要改。",
    "",
    "## 搜索",
    "需要时效信息（新版本、活动时间）时再用搜索，日常闲聊不查。",
    "",
    "## 兴趣与知识",
    "音游、射击游戏、米游剧情、声优梗、魔法少女题材、甜食。",
    "对政治、真实身份、群内站队没有兴趣，也不参与。",
]
PERSONA_FILLER = "群友闲聊时保持轻松，别抢话，也别把话题拽回自己身上。"

DIGEST_TEXT = (
    "身份：伊莉雅，游戏群里陪聊的魔法少女群友。\n"
    "会接的话题：音游、射击游戏、米游剧情、声优梗、魔法少女题材、甜食。\n"
    "雷区：政治、真实身份、群内站队、别人的隐私、说教式安慰。\n"
    "长度上限：一次最多 40 字、两句。"
)


def build_persona_text(target_len: int = PERSONA_TARGET_LEN) -> str:
    text = "\n".join(PERSONA_SECTIONS)
    while len(text) < target_len:
        text += "\n" + PERSONA_FILLER
    return text[:target_len]


PERSONA_TEXT = build_persona_text()


@pytest.fixture(autouse=True)
def _reset_logger():
    """logger 是共享 MagicMock；每个用例先清空调用记录，便于断言 warning 次数。"""
    astrbot_logger.reset_mock()
    yield


def _warnings_containing(keyword: str) -> list[str]:
    messages = []
    for call in astrbot_logger.warning.call_args_list:
        text = " ".join(str(arg) for arg in call.args)
        if keyword in text:
            messages.append(text)
    return messages


def _decision_json(should_reply: bool = True) -> str:
    return json.dumps(
        {
            "should_reply": should_reply,
            "is_questioned": False,
            "is_interesting": True,
            "reply_strategy": "接一句",
            "topic": "今晚打游戏",
            "reply_target": "小红",
            "entities": ["小红"],
            "facts": ["小红 问 今晚打不打"],
            "keywords": ["游戏"],
        },
        ensure_ascii=False,
    )


class RecordingProvider:
    """记录每次判定/生成请求的 provider 桩。"""

    def __init__(self, completion_text: str | None = None, error: Exception | None = None):
        self.requests: list[dict] = []
        self.completion_text = completion_text or _decision_json()
        self.error = error

    async def text_chat(self, *args, **kwargs):
        self.requests.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(completion_text=self.completion_text)


class FakePersonaManager:
    """按 umo 返回固定人格；resolve_selected_persona 只走一条链。"""

    def __init__(self, prompts: dict[str, str], mapping: dict[str, str] | None = None):
        self.prompts = prompts
        self.mapping = mapping or {CHAT_ID: PERSONA_NAME}

    async def resolve_selected_persona(
        self, umo, conversation_persona_id=None, platform_name=None, **_kwargs
    ):
        name = self.mapping.get(str(umo), PERSONA_NAME)
        prompt = self.prompts.get(name, "")
        if not prompt:
            return None, None, None, False
        return name, {"name": name, "prompt": prompt}, None, False

    async def get_all_personas(self):
        return [
            types.SimpleNamespace(persona_id=name, system_prompt=prompt)
            for name, prompt in self.prompts.items()
        ]


def build_config_manager(mode: str = "full", model: str = "fake-analyzer") -> ConfigManager:
    return ConfigManager(
        {
            "analyzer_model": model,
            "is_reasoning_model": True,
            "analyzer_persona_mode": mode,
        }
    )


def build_analyzer(
    provider,
    mode: str = "full",
    store: DigestStore | None = None,
    config_manager: ConfigManager | None = None,
    persona_manager: FakePersonaManager | None = None,
) -> LLMAnalyzer:
    context = types.SimpleNamespace(
        persona_manager=persona_manager or FakePersonaManager({PERSONA_NAME: PERSONA_TEXT}),
        conversation_manager=None,
        get_provider_by_id=lambda _name: provider,
    )
    cm = config_manager or build_config_manager(mode)
    return LLMAnalyzer("fake-analyzer", context, cm, store)


HISTORY = [
    {
        "role": "user",
        "content": "昨天那个boss真难",
        "sender_name": "小明",
        "sender_id": "1001",
        "timestamp": 1700000000.0,
    }
]
RECENT = [
    {
        "role": "user",
        "content": "今晚还打吗",
        "sender_name": "小红",
        "sender_id": "1002",
        "timestamp": 1700000100.0,
    }
]


async def run_analysis(analyzer: LLMAnalyzer):
    return await analyzer.analyze_and_decide(
        historical_context=HISTORY,
        recent_dialogue=RECENT,
        chat_id=CHAT_ID,
    )


# ---------------------------------------------------------------------------
# 验收 1 / 2：判定请求体
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_mode_keeps_persona_in_system(tmp_path):
    """默认 full：system 就是人格全文，长度与人格卡一致（≈5270-5290 量级）。"""
    provider = RecordingProvider()
    analyzer = build_analyzer(provider, mode="full")

    decision = await run_analysis(analyzer)

    assert decision.should_reply is True
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request["system_prompt"] == PERSONA_TEXT
    assert 5270 <= len(request["system_prompt"]) <= 5290
    # 两条用户消息：判定正文 + 对话记录
    assert request["contexts"][0]["role"] == "user"
    assert "<未回应消息>" in request["prompt"]


@pytest.mark.asyncio
async def test_digest_mode_system_only_has_digest_and_user_messages_unchanged(tmp_path):
    """digest：system ≤400 字符，两条 user 消息与 full 模式逐字节一致，JSON 仍可解析。"""
    store = DigestStore(str(tmp_path))
    store.set(PERSONA_NAME, DIGEST_TEXT, "auto", "fake-analyzer")

    full_provider = RecordingProvider()
    await run_analysis(build_analyzer(full_provider, mode="full"))

    digest_provider = RecordingProvider()
    decision = await run_analysis(
        build_analyzer(digest_provider, mode="digest", store=store)
    )

    full_request = full_provider.requests[0]
    digest_request = digest_provider.requests[0]

    assert digest_request["system_prompt"] == DIGEST_TEXT
    assert len(digest_request["system_prompt"]) <= 400
    assert digest_request["system_prompt"] != full_request["system_prompt"]
    # 只换 system：user#1 / user#2 字节级不变
    assert digest_request["contexts"] == full_request["contexts"]
    assert digest_request["prompt"] == full_request["prompt"]
    assert decision.should_reply is True
    assert decision.topic == "今晚打游戏"


@pytest.mark.asyncio
async def test_digest_missing_falls_back_to_full_once_per_card(tmp_path):
    """digest 但摘要缺失：回落人格全文；同一张卡每次 reload 只 warning 一次。"""
    provider = RecordingProvider()
    analyzer = build_analyzer(provider, mode="digest", store=DigestStore(str(tmp_path)))

    await run_analysis(analyzer)
    await run_analysis(analyzer)

    assert [r["system_prompt"] for r in provider.requests] == [PERSONA_TEXT, PERSONA_TEXT]
    warnings = _warnings_containing("判定摘要缺失")
    assert len(warnings) == 1, warnings

    # reload 后允许重新提示一次
    analyzer.reload_config(build_config_manager("digest"))
    await run_analysis(analyzer)
    assert len(_warnings_containing("判定摘要缺失")) == 2


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_uses_fixed_instruction_and_validates_length(tmp_path):
    """抽取指令原样传给 prompt；system_prompt=None；人格全文放 contexts。"""
    provider = RecordingProvider(completion_text=DIGEST_TEXT)

    digest = await generate(provider, PERSONA_TEXT, "fake-analyzer")

    assert digest == DIGEST_TEXT
    request = provider.requests[0]
    assert request["prompt"] == DIGEST_INSTRUCTION
    assert request["system_prompt"] is None
    assert request["contexts"] == [{"role": "user", "content": PERSONA_TEXT}]
    # 指令本身必须覆盖四行标签与长度约束
    for keyword in ("身份：", "会接的话题：", "雷区：", "长度上限：", "400"):
        assert keyword in DIGEST_INSTRUCTION


@pytest.mark.asyncio
async def test_generate_rejects_out_of_range_output():
    """太短/太长都丢弃并记 warning，不返回半成品。"""
    short = await generate(RecordingProvider(completion_text="身份：测试"), PERSONA_TEXT, "m")
    assert short is None

    long_text = "身" * (DIGEST_MAX_CHARS + 1)
    too_long = await generate(RecordingProvider(completion_text=long_text), PERSONA_TEXT, "m")
    assert too_long is None

    assert len(_warnings_containing("长度不合规")) == 2


@pytest.mark.asyncio
async def test_generate_handles_missing_provider_and_errors():
    assert await generate(None, PERSONA_TEXT, "m") is None
    failing = RecordingProvider(error=RuntimeError("boom"))
    assert await generate(failing, PERSONA_TEXT, "m") is None
    assert len(_warnings_containing("生成调用失败")) == 1


# ---------------------------------------------------------------------------
# ensure_all：串行补生成
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_all_generates_only_missing_cards(tmp_path, monkeypatch):
    """同一张卡只生成一次；已有摘要的卡跳过；落盘 source=auto / model。"""
    import astrbot_plugin_angel_heart.core.persona_digest as persona_digest_module

    monkeypatch.setattr(persona_digest_module, "GENERATE_INTERVAL_SECONDS", 0)
    store = DigestStore(str(tmp_path))
    store.set(OTHER_PERSONA, DIGEST_TEXT, "manual", "old-model")

    provider = RecordingProvider(completion_text=DIGEST_TEXT)
    prompts = {PERSONA_NAME: PERSONA_TEXT, OTHER_PERSONA: "美游的人格卡正文"}
    mapping = {
        "default:GroupMessage:10001": PERSONA_NAME,
        "default:GroupMessage:10002": PERSONA_NAME,  # 同一张卡的两个会话
        "default:GroupMessage:10003": OTHER_PERSONA,
    }
    persona_manager = FakePersonaManager(prompts, mapping)

    async def resolver(umo: str):
        name, persona, _, _ = await persona_manager.resolve_selected_persona(umo=umo)
        return name, (persona or {}).get("prompt", "")

    await ensure_all(
        build_config_manager(),
        lambda _name: provider,
        resolver,
        list(mapping.keys()),
        store,
    )

    assert len(provider.requests) == 1  # 只有伊莉雅缺摘要
    entry = store.entries()[PERSONA_NAME]
    assert entry["digest"] == DIGEST_TEXT
    assert entry["source"] == "auto"
    assert entry["model"] == "fake-analyzer"
    assert entry["generated_at"]
    assert store.entries()[OTHER_PERSONA]["source"] == "manual"

    # 再跑一次：全部命中已有摘要，不再调用模型
    await ensure_all(
        build_config_manager(), lambda _name: provider, resolver, list(mapping.keys()), store
    )
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_ensure_all_provider_missing_warns_once_and_keeps_store_clean(tmp_path):
    """秘书模型不存在：warning 只一条（不按卡刷屏），不落条目。"""
    store = DigestStore(str(tmp_path))
    prompts = {PERSONA_NAME: PERSONA_TEXT, OTHER_PERSONA: "美游的人格卡正文"}
    mapping = {
        "default:GroupMessage:10001": PERSONA_NAME,
        "default:GroupMessage:10002": OTHER_PERSONA,
    }
    persona_manager = FakePersonaManager(prompts, mapping)

    async def resolver(umo: str):
        name, persona, _, _ = await persona_manager.resolve_selected_persona(umo=umo)
        return name, (persona or {}).get("prompt", "")

    await ensure_all(
        build_config_manager(), lambda _name: None, resolver, list(mapping.keys()), store
    )

    assert store.entries() == {}
    warnings = _warnings_containing("不可用")
    assert len(warnings) == 1, warnings
    assert "2 张" in warnings[0]


@pytest.mark.asyncio
async def test_ensure_all_failure_does_not_overwrite_existing(tmp_path, monkeypatch):
    """生成失败：记 warning，不覆盖已有摘要、不新增条目。"""
    import astrbot_plugin_angel_heart.core.persona_digest as persona_digest_module

    monkeypatch.setattr(persona_digest_module, "GENERATE_INTERVAL_SECONDS", 0)
    store = DigestStore(str(tmp_path))
    failing = RecordingProvider(error=RuntimeError("model down"))
    persona_manager = FakePersonaManager({PERSONA_NAME: PERSONA_TEXT})

    async def resolver(umo: str):
        name, persona, _, _ = await persona_manager.resolve_selected_persona(umo=umo)
        return name, (persona or {}).get("prompt", "")

    await ensure_all(
        build_config_manager(), lambda _name: failing, resolver, [CHAT_ID], store
    )

    assert store.get(PERSONA_NAME) == ""
    assert _warnings_containing("生成调用失败")


# ---------------------------------------------------------------------------
# 摘要存储
# ---------------------------------------------------------------------------


def test_digest_store_roundtrip_and_missing(tmp_path):
    """落盘结构 version=1；空 digest 视为待生成；reset 删条目。"""
    store = DigestStore(str(tmp_path))
    assert store.missing([PERSONA_NAME]) == [PERSONA_NAME]

    entry = store.set(PERSONA_NAME, DIGEST_TEXT, "manual", "m1")
    assert entry["source"] == "manual"
    assert store.get(PERSONA_NAME) == DIGEST_TEXT
    assert store.missing([PERSONA_NAME]) == []

    reloaded = DigestStore(str(tmp_path))
    assert reloaded.get(PERSONA_NAME) == DIGEST_TEXT

    payload = json.loads((tmp_path / "analyzer_personas.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert set(payload["personas"][PERSONA_NAME]) == {
        "digest",
        "generated_at",
        "model",
        "source",
    }

    # 空串 = 待生成
    store.set(PERSONA_NAME, "   ", "manual")
    assert store.missing([PERSONA_NAME]) == [PERSONA_NAME]

    assert store.reset(PERSONA_NAME) is True
    assert store.reset(PERSONA_NAME) is False
    assert store.entries() == {}


def test_collect_service_umos_unions_whitelist_and_bindings(tmp_path):
    """会话范围 = 绑定 ∪ 白名单；纯群号用来源登记展开成完整 origin。"""

    class FakeProfileStore:
        def list_bindings(self):
            return [{"chat_id": "default:GroupMessage:20002", "template_id": "tpl_1"}]

    class FakeSources:
        def list_sources(self):
            return [
                {"chat_id": "default:GroupMessage:10001", "kind": "group"},
                {"chat_id": "default:FriendMessage:10001", "kind": "private"},
            ]

    manager = ConfigManager({"access_control": {"chat_ids": ["10001", "20002", "30003"]}})

    umos = collect_service_umos(manager, FakeProfileStore(), FakeSources())
    assert umos == [
        "default:GroupMessage:20002",  # 绑定优先，白名单里的 20002 去重
        "default:GroupMessage:10001",  # 纯群号展开成群聊 origin（不串到同号私聊）
        "30003",  # 没有来源登记时保留纯群号
    ]


# ---------------------------------------------------------------------------
# 配置与 schema
# ---------------------------------------------------------------------------


def test_persona_mode_property_falls_back_to_full():
    assert ConfigManager({"analyzer_persona_mode": "digest"}).analyzer_persona_mode == "digest"
    assert ConfigManager({"analyzer_persona_mode": "DIGEST "}).analyzer_persona_mode == "digest"
    assert ConfigManager({"analyzer_persona_mode": "bogus"}).analyzer_persona_mode == "full"
    assert ConfigManager({}).analyzer_persona_mode == "full"


def test_conf_schema_declares_persona_mode():
    import pathlib

    schema_path = pathlib.Path(__file__).resolve().parent.parent / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    meta = schema["analyzer_persona_mode"]
    assert meta["type"] == "string"
    assert meta["default"] == "full"
    assert meta["options"] == ["full", "digest"]
    assert meta["description"] == "判定人格摘要"


# ---------------------------------------------------------------------------
# Web API
# ---------------------------------------------------------------------------


class WebFakeContext:
    def __init__(self, persona_manager=None):
        self.routes = []
        self.persona_manager = persona_manager

    def register_web_api(self, route, handler, methods, desc):
        self.routes.append((route, handler, methods, desc))


def _build_quart_app(fake_context):
    from quart import Quart

    app = Quart(__name__)
    for route, handler, methods, _ in fake_context.routes:
        path = "/api/plug" + route
        app.add_url_rule(path, endpoint=path, view_func=handler, methods=methods)
    return app


def build_persona_api_app(tmp_path, mode="digest"):
    from astrbot_plugin_angel_heart.core.chat_sources import ChatSourcesStore

    import web_api as web_api_module

    store = DigestStore(str(tmp_path))
    manager = build_config_manager(mode)
    manager._config["access_control"] = {"chat_ids": ["10001"]}
    sources = ChatSourcesStore(str(tmp_path))
    sources.record(CHAT_ID, "测试群", "group")
    persona_manager = FakePersonaManager(
        {PERSONA_NAME: PERSONA_TEXT, OTHER_PERSONA: "美游的人格卡正文"}
    )

    async def resolver(umo: str):
        name, persona, _, _ = await persona_manager.resolve_selected_persona(umo=umo)
        return name, (persona or {}).get("prompt", "")

    class FakeProfileStore:
        def list_bindings(self):
            return []

    fake_context = WebFakeContext(persona_manager)
    web_api_module.register_all_routes(
        fake_context,
        FakeProfileStore(),
        manager,
        type("Ledger", (), {"get_all_chat_ids": staticmethod(lambda: [])})(),
        chat_sources=sources,
        persona_digest_store=store,
        persona_resolver=resolver,
    )
    return _build_quart_app(fake_context), store


@pytest.mark.asyncio
async def test_api_lists_in_use_and_other_personas(tmp_path):
    """列表：mode/秘书模型 + 在用（带 used_by）+ 其他全量卡。"""
    app, store = build_persona_api_app(tmp_path, mode="digest")
    store.set(PERSONA_NAME, DIGEST_TEXT, "manual", "fake-analyzer")

    client = app.test_client()
    resp = await client.get("/api/plug/astrbot_plugin_angel_heart/analyzer_personas")
    data = await resp.get_json()

    assert data["status"] == "ok"
    payload = data["data"]
    assert payload["mode"] == "digest"
    assert payload["analyzer_model"] == "fake-analyzer"
    by_name = {item["name"]: item for item in payload["personas"]}
    assert set(by_name) == {PERSONA_NAME, OTHER_PERSONA}

    in_use = by_name[PERSONA_NAME]
    assert in_use["digest"] == DIGEST_TEXT
    assert in_use["source"] == "manual"
    assert in_use["used_by"] == [{"umo": CHAT_ID, "display_name": "测试群"}]

    other = by_name[OTHER_PERSONA]
    assert other["used_by"] == []
    assert other["digest"] == ""


@pytest.mark.asyncio
async def test_api_save_and_reset(tmp_path):
    """保存写 manual；超 600 字符拒绝；清空删条目并提示重载。"""
    app, store = build_persona_api_app(tmp_path)
    client = app.test_client()

    resp = await client.post(
        "/api/plug/astrbot_plugin_angel_heart/analyzer_personas/save",
        json={"name": PERSONA_NAME, "digest": DIGEST_TEXT},
    )
    data = await resp.get_json()
    assert data["status"] == "ok"
    assert data["data"]["source"] == "manual"
    assert data["data"]["generated_at"]
    assert store.get(PERSONA_NAME) == DIGEST_TEXT

    resp = await client.post(
        "/api/plug/astrbot_plugin_angel_heart/analyzer_personas/save",
        json={"name": PERSONA_NAME, "digest": "长" * (DIGEST_MAX_CHARS + 1)},
    )
    assert resp.status_code == 400
    assert (await resp.get_json())["status"] == "error"
    assert store.get(PERSONA_NAME) == DIGEST_TEXT  # 未被覆盖

    resp = await client.post(
        "/api/plug/astrbot_plugin_angel_heart/analyzer_personas/reset",
        json={"name": PERSONA_NAME},
    )
    data = await resp.get_json()
    assert data["status"] == "ok"
    assert data["data"]["removed"] is True
    assert "重载插件" in data["message"]
    assert store.get(PERSONA_NAME) == ""

    resp = await client.post(
        "/api/plug/astrbot_plugin_angel_heart/analyzer_personas/save",
        json={"digest": DIGEST_TEXT},
    )
    assert (await resp.get_json())["status"] == "error"


# ---------------------------------------------------------------------------
# 验收 5：生成失败后判定照常（fail-open）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_failure_then_judgment_still_answers(tmp_path, monkeypatch):
    """provider 生成失败 → 不落条目；随后 digest 模式判定回落全文并正常出结果。"""
    import astrbot_plugin_angel_heart.core.persona_digest as persona_digest_module

    monkeypatch.setattr(persona_digest_module, "GENERATE_INTERVAL_SECONDS", 0)
    store = DigestStore(str(tmp_path))
    persona_manager = FakePersonaManager({PERSONA_NAME: PERSONA_TEXT})

    async def resolver(umo: str):
        name, persona, _, _ = await persona_manager.resolve_selected_persona(umo=umo)
        return name, (persona or {}).get("prompt", "")

    # 秘书模型名配错：拿不到 provider
    broken_manager = build_config_manager("digest", model="not-exist")
    await ensure_all(
        broken_manager, lambda _name: None, resolver, [CHAT_ID], store
    )
    assert store.entries() == {}
    assert len(_warnings_containing("不可用")) == 1

    # 判定链路本身仍能工作：摘要缺失回落全文，JSON 正常解析
    provider = RecordingProvider()
    analyzer = build_analyzer(
        provider, mode="digest", store=store, config_manager=build_config_manager("digest")
    )
    decision = await run_analysis(analyzer)

    assert decision.should_reply is True
    assert provider.requests[0]["system_prompt"] == PERSONA_TEXT
    assert len(_warnings_containing("判定摘要缺失")) == 1
