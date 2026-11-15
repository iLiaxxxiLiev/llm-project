"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.schemas import ENTITY_TYPES, EXTRACTION_SCHEMA, RELATION_SIGNATURES, Entity, Relation
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, ExtractionResult


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю
    # raise NotImplementedError("Реализуйте extract_chunk")

    names = vocabulary()
    directions = "\n".join(
        f"{subject_type} --{relation_type}--> {object_type}"
        for relation_type, (subject_type, object_type) in RELATION_SIGNATURES.items()
    )
    instruction = (
        "Извлеки из фрагмента сервисного документа сущности и связи.\n"
        "Включай только факты из фрагмента. Соблюдай направление каждой связи.\n"
        f"Допустимые типы сущностей: {', '.join(ENTITY_TYPES)}.\n"
        f"Допустимые направления связей:\n{directions}\n"
        f"Используй словарь для написания названий: {json.dumps(names, ensure_ascii=False)}.\n"
        "Связь FAILED обозначает отказ узла, зафиксированный в отчёте о выезде.\n"
        "Связь REPLACED_BY направлена от снятой с производства детали к её замене.\n"
        "Верни JSON с массивами entities и relations. При отсутствии фактов верни пустые массивы.\n"
        "Пример фрагмента: «Ошибка E-17 указывает на отказ пускового реле».\n"
        'Пример ответа: {"entities": [{"type": "ErrorCode", "name": "E-17"}, '
        '{"type": "Component", "name": "пусковое реле"}], '
        '"relations": [{"subject_type": "ErrorCode", "subject": "E-17", "type": "INDICATES", '
        '"object_type": "Component", "object": "пусковое реле"}]}'
    )


    if chunk.doc_id.startswith("procedure_"):
        instruction += (
            "\nДля каждого кода ошибки, к которому явно применяется регламент, добавь связь FIXED_BY. "
            "В subject запиши код ошибки, в subject_type — ErrorCode. "
            "В object запиши код регламента из заголовка, в object_type — Procedure."
        )

    result = ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
    correction = "\nИсправь формат ответа. Верни JSON по заданной схеме."

    for suffix in ("", correction):
        response = llm.chat(
            [
                {"role": "system", "content": instruction + suffix},
                {"role": "user", "content": chunk.text},
            ],
            json_schema=EXTRACTION_SCHEMA,
            max_tokens=4096,
        )
        try:
            payload = json.loads(response.content)
        except (TypeError, ValueError, RecursionError):
            continue

        if _matches_extraction_schema(payload, EXTRACTION_SCHEMA):
            break
    else:
        return result

    chunk_codes = set(find_codes(chunk.text))
    entities: dict[str, Entity] = {}

    def confirmed_entity(entity_type: str, raw_name: str) -> Entity | None:
        name = canonical_name(entity_type, raw_name, names)
        if name is None or (entity_type in CODE_TYPES and name not in chunk_codes):
            return None
        return Entity(type=entity_type, name=name)

    for row in payload["entities"]:
        entity = confirmed_entity(row["type"], row["name"])
        if entity is None:
            result.dropped += 1
            continue
        entities.setdefault(entity.node_id, entity)

    for row in payload["relations"]:
        subject = confirmed_entity(row["subject_type"], row["subject"])
        target = confirmed_entity(row["object_type"], row["object"])
        if subject is None or target is None:
            result.dropped += 1
            continue

        relation = Relation(subject=subject, type=row["type"], object=target)
        if not relation.matches_schema():
            result.dropped += 1
            continue

        result.relations.append(relation)
        entities.setdefault(subject.node_id, subject)
        entities.setdefault(target.node_id, target)

    if chunk.doc_id.startswith("manual_") and "## Состав узлов" in chunk.text:
        title, _, body = chunk.text.partition("\n")
        model_codes = find_codes(title)
        model = confirmed_entity("Model", model_codes[0]) if len(model_codes) == 1 else None

        if model is not None:
            existing_relations = set(result.relations)
            in_parts_section = False

            for raw_line in body.splitlines():
                line = raw_line.strip()
                if line.startswith("## "):
                    in_parts_section = line == "## Состав узлов"
                    continue
                if not in_parts_section or not line.startswith("- "):
                    continue

                component_name, separator, details = line[2:].partition(" – деталь ")
                if not separator:
                    continue

                part_code = details.partition(",")[0]
                part = confirmed_entity("Part", part_code)
                if part is None:
                    continue

                manual_relations = [Relation(subject=model, type="HAS_PART", object=part)]
                component = confirmed_entity("Component", component_name)
                if component is not None:
                    manual_relations.append(Relation(subject=part, type="IS_A", object=component))

                for relation in manual_relations:
                    if relation in existing_relations:
                        continue

                    result.relations.append(relation)
                    existing_relations.add(relation)
                    entities.setdefault(relation.subject.node_id, relation.subject)
                    entities.setdefault(relation.object.node_id, relation.object)

    result.entities = list(entities.values())
    return result


def _matches_extraction_schema(payload: object, schema: dict) -> bool:
    """Проверяет массивы, обязательные поля записей и допустимые типы."""
    if not isinstance(payload, dict):
        return False

    for key in schema["required"]:
        rows = payload.get(key)
        if not isinstance(rows, list):
            return False

        item_schema = schema["properties"][key]["items"]
        for row in rows:
            if not isinstance(row, dict):
                return False

            for field in item_schema["required"]:
                value = row.get(field)
                allowed = item_schema["properties"][field].get("enum")
                if not isinstance(value, str) or (allowed is not None and value not in allowed):
                    return False

    return True