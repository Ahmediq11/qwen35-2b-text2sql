"""Prompt templates, one per model family (PLAN §4.2).

qwen   : user message = schema + question (+ BIRD evidence) + the fixed instruction; same for every Qwen3.5 model.
xiyan  : XiYanSQL's own documented template (nl2sqlite_template_cn, copied verbatim from the model card at revision
         b883e58ed83f74bab037d6a7b90c4b8706d357d7); the schema is the same DDL + 3 sample rows text (the card reports
         DDL results too: BIRD dev 52.09, Spider test 80.57). Its answer continues after the closing "```sql".
"""
from data import schema_text

INSTRUCTION = "Write one SQLite query that answers the question. Put the final query in a ```sql code block."

XIYAN_TEMPLATE = (
    "你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，"
    "并运用{dialect}知识生成sql语句回答【用户问题】。\n"
    "【用户问题】\n{question}\n\n"
    "【数据库schema】\n{db_schema}\n\n"
    "【参考信息】\n{evidence}\n\n"
    "【用户问题】\n{question}\n\n"
    "```sql")


def qwen_user(item, sample_rows=3):
    parts = ["Database schema:", schema_text(item["db_path"], sample_rows), "", f"Question: {item['question']}"]
    if item.get("evidence"):
        parts.append(f"Evidence: {item['evidence']}")
    parts += ["", INSTRUCTION]
    return "\n".join(parts)


def messages(item, family):
    if family == "qwen":
        return [{"role": "user", "content": qwen_user(item)}]
    if family == "xiyan":
        return [{"role": "user", "content": XIYAN_TEMPLATE.format(
            dialect="SQLite", question=item["question"], db_schema=schema_text(item["db_path"]),
            evidence=item.get("evidence") or "")}]
    raise ValueError(family)


def answer_text(raw_output, family):
    """make every family's output look like '... ```sql <query> ```' for the shared extractor."""
    if family == "xiyan" and "```sql" not in raw_output.lower():
        # the prompt already opened the ```sql fence: the model writes the query and maybe closes the fence
        body = raw_output.split("```", 1)[0]
        return "```sql\n" + body.strip() + "\n```"
    return raw_output
