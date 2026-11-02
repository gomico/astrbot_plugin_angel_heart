# 秘书分析提示词

秘书分析请求由三部分组成，都在 `prompts/`：

- `system_prompt.md`：群聊参与判定
- `output_contract.md`：输出字段与格式，固定不变
- `conversation_template.md`：待分析的对话记录模板，固定不变

人格不再由插件自建。会话人格来自 AstrBot 人格系统，作为分析请求的 **system**；插件规则作为 **第一条用户消息**。

## 组装方式

`LLMAnalyzer._build_prompts` 产出两条用户消息：

1. **规则文本**：参与判定 + 输出契约。判定正文按 `PromptSettingsStore` 二选一：预设为 `system_prompt.md` 全文，自定义为用户填写内容；自定义为空时回落预设。`{alias}` 按该群配置替换。
2. **对话记录**：把历史上下文和最近对话填进 `conversation_template.md`，分别包在 `<已回应消息>` / `<未回应消息>`。若有工作账本，以第三人称追加在对话记录后，提醒秘书不要让助理处理重复问题。

`LLMAnalyzer._call_ai_model` 再把会话人格放进 `system_prompt`，把规则文本放进 `contexts` 的第一条用户消息，对话记录作为当前 `prompt`。

## 输出契约

秘书必须产出 `SecretaryDecision`：

- `should_reply`
- `is_questioned`
- `is_interesting`
- `reply_strategy`
- `topic`
- `reply_target`
- `entities` / `facts` / `keywords`

这些字段给旁路注入和检索线索用，不进主脑请求体。

## 判定人格摘要（`analyzer_persona_mode`）

`analyzer_persona_mode` 决定秘书判定 system 的来源，只影响 system，不动两条用户消息：

- `full`（默认）：会话人格全文，与历史行为完全一致。
- `digest`：每张人格卡一份「判定用摘要」（四行、合计 ≤400 字）。只保留会影响判断的
  身份、会接的话题、雷区、长度上限；人格卡里的工具、生图、搜索、技能、记忆段落，
  以及说话风格与示例都不进 system。

摘要按人格卡落盘到插件数据目录 `analyzer_personas.json`，key 是人物的 `name`
（AstrBot v3 的 `persona_id` 就是 `name`）：

```json
{
  "version": 1,
  "personas": {
    "<persona_name>": {
      "digest": "四行摘要文本",
      "generated_at": "2026-11-02T12:00:00+08:00",
      "model": "<analyzer_model>",
      "source": "auto"
    }
  }
}
```

规则：

- 条目缺失或 `digest` 为空串都算「待生成」；只存摘要，不缓存人格原文、不做 hash
  自动失效、不做定时重生成，用户大改人格后自己清空。
- 生成走秘书模型（`analyzer_model`）的 `text_chat`：人格全文放 `contexts`，抽取指令
  作 `prompt`，`system_prompt=None`；输出必须在 80–600 字符之间，否则丢弃并记 warning。
- 生成时机：插件初始化（含 reload）与 WebUI 保存配置后，后台任务枚举本插件服务的
  会话（`access_control.chat_ids` ∪ 已绑定模板的会话），解析各自生效人格，对缺失的
  卡串行生成（每张间隔 0.5s，同一张卡一次 reload 内只生成一次），不阻塞加载与判定。
- 失败与生成期间都 fail-open：判定回落人格全文，warning 每张卡每次 reload 只一条。
- 设置页「判定人格摘要」可查看在用 / 其他卡片、被哪些会话使用，并手工保存或清空。

