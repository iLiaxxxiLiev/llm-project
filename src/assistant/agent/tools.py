"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
TOOL_SCHEMAS: list[dict] = []


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    # raise NotImplementedError("Реализуйте build_tools")

    from assistant.graphrag.search import global_search, local_search

    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Передайте непустой вопрос для поиска."}

        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Передайте непустой вопрос для поиска."}

        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        if not _valid_part_id(part_id):
            return {"error": _INVALID_PART}
        if warehouse not in _WAREHOUSES:
            return {"error": _INVALID_WAREHOUSE}

        return _call_erp(erp.get_stock, part_id=part_id, warehouse=warehouse)

    def get_price(part_id: str) -> dict:
        if not _valid_part_id(part_id):
            return {"error": _INVALID_PART}

        return _call_erp(erp.get_price, part_id=part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        if not _valid_part_id(part_id):
            return {"error": _INVALID_PART}
        if warehouse not in _WAREHOUSES:
            return {"error": _INVALID_WAREHOUSE}
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": "Количество должно быть целым числом больше нуля."}
        if not isinstance(model, str) or not model.strip():
            return {"error": "Укажите модель оборудования."}

        return _call_erp(erp.create_order, part_id=part_id, warehouse=warehouse, qty=qty, model=model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }


_WAREHOUSES = ("Москва", "Санкт-Петербург", "Казань", "Новосибирск")
_PART_PATTERN = r"^P-[0-9]{4}[A-Z]?$"
_PART_ID = {"type": "string", "pattern": _PART_PATTERN, "description": "Номер детали, например P-1042B."}
_WAREHOUSE = {"type": "string", "enum": list(_WAREHOUSES), "description": "Название склада."}
_QUESTION = {"type": "string", "minLength": 1, "description": "Вопрос для поиска по документам."}
_INVALID_PART = "Укажите номер детали в формате P-1042 или P-1042B."
_INVALID_WAREHOUSE = f"Неизвестный склад. Допустимые склады — {', '.join(_WAREHOUSES)}."


def _valid_part_id(value: object) -> bool:
    import re

    return isinstance(value, str) and re.fullmatch(_PART_PATTERN, value) is not None


def _call_erp(method: Callable[..., dict], **arguments) -> dict:
    from assistant.infra.erp_client import ErpNotFound, ErpRejected

    try:
        return method(**arguments)
    except ErpNotFound as exc:
        return {"error": f"Данные в ERP не найдены. {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос. {exc}"}


def _tool_schema(name: str, description: str, **properties: dict) -> dict:
    """Собирает описание функции для модели."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS.extend(
    [
        _tool_schema(
            "search_knowledge",
            "Найти факты, источники и актуальные замены для конкретной модели, детали, ошибки или регламента.",
            question=_QUESTION,
        ),
        _tool_schema(
            "summarize_topic",
            "Ответить на обзорный вопрос по сводкам сообществ графа знаний.",
            question=_QUESTION,
        ),
        _tool_schema(
            "check_stock",
            "Узнать остаток детали на указанном складе и её статус в ERP.",
            part_id=_PART_ID,
            warehouse=_WAREHOUSE,
        ),
        _tool_schema(
            "get_price",
            "Узнать цену детали в рублях и её статус в ERP.",
            part_id=_PART_ID,
        ),
        _tool_schema(
            "create_service_order",
            "Создать сервисный заказ на деталь в ERP и получить номер заказа.",
            part_id=_PART_ID,
            warehouse=_WAREHOUSE,
            qty={"type": "integer", "minimum": 1, "description": "Количество деталей в заказе."},
            model={"type": "string", "minLength": 1, "description": "Модель оборудования, например AR-210."},
        ),
    ]
)
