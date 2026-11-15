"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

from collections.abc import Callable

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import AgentResult


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа
    # raise NotImplementedError("Реализуйте run_agent")

    import json

    from assistant.agent.tools import TOOL_SCHEMAS
    from assistant.infra.grounding import find_codes
    from assistant.infra.llm import tool_message
    from assistant.infra.schemas import ToolStep

    schemas = {entry["function"]["name"]: entry["function"]["parameters"] for entry in TOOL_SCHEMAS}
    messages = [
        {
            "role": "system",
            "content": (
                "Ты помощник диспетчера сервисной службы «ПолярТех». Отвечай по-русски и используй инструменты. "
                "Для точных вопросов по документации вызывай search_knowledge, для обзоров — summarize_topic. "
                "Чтобы узнать деталь по модели и коду ошибки, передай оба кода в search_knowledge. "
                "Если у детали есть замена, используй последний номер в цепочке. "
                "Проверяй цену и остаток отдельно через get_price и check_stock. "
                "Для каждого вызова ERP бери номер детали из вопроса или полученных фактов. "
                "Вызывай create_service_order при явной просьбе оформить заказ и указанном количестве. "
                "Вопрос «какую деталь заказать» просит подобрать деталь. "
                "Указывай номер только у заказа, подтверждённого результатом инструмента. "
                f"Если сведений для ответа нет, напиши «{_NOT_FOUND}»."
            ),
        },
        {"role": "user", "content": question},
    ]
    result = AgentResult(answer="")
    part_codes = set(find_codes(question))
    quantities = _numbers(_QUANTITY, question)
    cache: dict[tuple[str, str], dict] = {}

    while len(result.steps) < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not response.tool_calls:
            return _finish(result, response.content, question, messages, llm)

        remaining = max_steps - len(result.steps)
        assistant_message = response.as_message()
        assistant_message["tool_calls"] = assistant_message["tool_calls"][:remaining]
        messages.append(assistant_message)

        for call in response.tool_calls[:remaining]:
            args = call.arguments
            step = ToolStep(tool=call.name, args=args)
            error = _argument_error(args, schemas.get(call.name)) if call.name in tools else "Неизвестный инструмент."
            if error is None and "part_id" in args and args["part_id"] not in part_codes:
                error = "Номер детали не подтверждён. Вызови search_knowledge, чтобы узнать номер по документам."

            if error is None and call.name == _ORDER_TOOL:
                if args["qty"] not in quantities:
                    step.error = "Количество заказа не подтверждено вопросом."
                    result.steps.append(step)
                    result.answer = "Сколько штук нужно заказать? Укажите количество."
                    return result
                if not confirm:
                    result.answer = (
                        f"Подтвердите заказ детали {args['part_id']} для {args['model']}. "
                        f"Количество деталей — {args['qty']}. Склад — {args['warehouse']}."
                    )
                    result.needs_confirmation = True
                    result.pending_action = {"tool": call.name, "args": args.copy()}
                    return result

            result.steps.append(step)
            if error is None:
                key = (call.name, json.dumps(args, ensure_ascii=False, sort_keys=True))
                if key not in cache:
                    cache[key] = _invoke(tools[call.name], args)
                payload = cache[key]
            else:
                payload = {"error": error}

            if "error" in payload:
                step.error = payload["error"]
            else:
                step.result = payload
                part_codes.update(find_codes(json.dumps(payload, ensure_ascii=False)))
                result.sources = sorted(set(result.sources) | set(payload.get("sources", [])))
            messages.append(tool_message(call, payload))

    result.answer = " ".join([*_observations(result), "Достигнут лимит вызовов инструментов. Работа прервана."])
    return result


_NOT_FOUND = "В базе знаний не найдено."
_ERP_UNAVAILABLE = "Данные ERP получить не удалось."
_ORDER_TOOL = "create_service_order"
_ORDER_ID = r"\bSO-[0-9]+\b"
_NUMBER = r"(?:[0-9]{1,3}(?:[ \u00a0\u202f][0-9]{3})+|[0-9]+)"
_LEFT = r"(?<![\w-])(?<![0-9][.,])"
_RIGHT = r"(?![\w-]|[.,][0-9])"
_PRICE = (
    rf"{_LEFT}({_NUMBER})\s*(?=₽|руб(?:\.|ля\b|лей\b|\b))"
    rf"|\b(?:цена|стоимость|стоит)\s*(?:составляет\s*)?(?:[:=–—-]\s*)?({_NUMBER}){_RIGHT}"
)
_STOCK = (
    rf"\b(?:остаток|в наличии|на складе)\b[^\d.,\n]{{0,45}}?{_LEFT}({_NUMBER}){_RIGHT}"
)
_QUANTITY_WORDS = {
    "один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
}
_COUNT = rf"(?:{_NUMBER}|{'|'.join(_QUANTITY_WORDS)})"
_QUANTITY = (
    rf"{_LEFT}({_COUNT})\s*(?=шт\b|штук(?:а|и)?\b|детал(?:ь|и|ей)\b|"
    r"единиц(?:а|ы)?\b|экземпляр(?:а|ов)?\b|[PР][- ][0-9]{4})"
    rf"|\b(?:количество|в количестве|закажи(?:те)?|оформи(?:те)? заказ(?: на)?)"
    rf"\s*[:=]?\s*({_COUNT}){_RIGHT}(?=\s*(?:$|[.,!?]|для\b|на склад\b))"
)


def _argument_error(args: dict, schema: dict | None) -> str | None:
    import re

    if schema is None:
        return "Неизвестный инструмент."
    if not isinstance(args, dict) or args.keys() != schema["properties"].keys():
        return "Передай все параметры инструмента по заданной схеме."

    for name, value in args.items():
        field = schema["properties"][name]
        if field["type"] == "integer":
            if isinstance(value, bool) or not isinstance(value, int) or value < field["minimum"]:
                return f"Аргумент {name} должен быть целым числом больше нуля."
        elif not isinstance(value, str) or not value.strip():
            return f"Аргумент {name} должен быть непустой строкой."
        elif "enum" in field and value not in field["enum"]:
            return f"Недопустимое значение аргумента {name}."
        elif "pattern" in field and not re.fullmatch(field["pattern"], value):
            return f"Неверный формат аргумента {name}."
    return None


def _invoke(tool: Callable[..., dict], args: dict) -> dict:
    from assistant.infra.erp_client import ErpUnavailable

    for _ in range(3):
        try:
            return tool(**args)
        except ErpUnavailable:
            continue
        except TypeError:
            return {"error": "Аргументы не подходят для вызова инструмента."}
    return {"error": _ERP_UNAVAILABLE}


def _numbers(pattern: str, text: str) -> set[int]:
    import re

    values = set()
    for match in re.finditer(pattern, text, re.IGNORECASE):
        raw = next(group for group in match.groups() if group is not None).lower()
        values.add(_QUANTITY_WORDS[raw] if raw in _QUANTITY_WORDS else int("".join(raw.split())))
    return values


def _order_ids(text: str) -> set[str]:
    import re

    return set(re.findall(_ORDER_ID, text.upper()))


def _unsupported(text: str, question: str, payloads: list[dict]) -> bool:
    import json

    from assistant.infra.grounding import find_codes

    evidence = question + "\n" + json.dumps(payloads, ensure_ascii=False)
    codes = set(find_codes(evidence))
    orders = {payload["order_id"].upper() for payload in payloads if payload.get("order_id")}
    prices = _numbers(_PRICE, question) | {payload["price_rub"] for payload in payloads if "price_rub" in payload}
    stock = _numbers(_STOCK, question) | {
        payload["qty"]
        for payload in payloads
        if "qty" in payload and "warehouse" in payload and "order_id" not in payload
    }
    quantities = stock | _numbers(_QUANTITY, question) | {
        payload["qty"] for payload in payloads if "qty" in payload and "order_id" in payload
    }
    return bool(
        set(find_codes(text)) - codes
        or _order_ids(text) - orders
        or _numbers(_PRICE, text) - prices
        or _numbers(_STOCK, text) - stock
        or _numbers(_QUANTITY, text) - quantities
    )


def _observations(result: AgentResult) -> list[str]:
    observations = []
    for step in result.steps:
        payload = step.result
        if payload and not payload.get("refused"):
            if step.tool in ("search_knowledge", "summarize_topic") and payload.get("answer"):
                observations.append(payload["answer"])
            elif step.tool == "check_stock":
                observations.append(f"{payload['part_id']}, склад {payload['warehouse']} — {payload['qty']} шт.")
            elif step.tool == "get_price":
                observations.append(f"{payload['part_id']} стоит {payload['price_rub']} руб.")
            elif step.tool == _ORDER_TOOL:
                observations.append(f"Заказ {payload['order_id']} создан.")
        elif step.error == _ERP_UNAVAILABLE:
            observations.append(_ERP_UNAVAILABLE)
    return list(dict.fromkeys(observations))


def _finish(result: AgentResult, draft: str, question: str, messages: list[dict], llm: LLMClient) -> AgentResult:
    payloads = [step.result for step in result.steps if step.result is not None]
    useful = any(not payload.get("refused") for payload in payloads)
    unavailable = any(step.error == _ERP_UNAVAILABLE for step in result.steps)
    no_data = any(payload.get("refused") for payload in payloads) or any(
        step.error and step.error.startswith("Данные в ERP не найдены.") for step in result.steps
    )

    if no_data and not useful and not unavailable:
        result.answer = _NOT_FOUND
        result.refused = True
        return result

    answer = draft.strip()
    unsupported = _unsupported(answer, question, payloads)
    if unsupported:
        correction = llm.chat(
            [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "Исправь последний ответ. Используй только коды, цены и остатки из исходного вопроса "
                        f"и результатов инструментов. Если данных недостаточно, напиши «{_NOT_FOUND}»."
                    ),
                },
            ]
        )
        answer = correction.content.strip()
        unsupported = bool(correction.tool_calls) or _unsupported(answer, question, payloads)

    orders = [payload["order_id"] for payload in payloads if payload.get("order_id")]
    if unsupported or not answer or answer.casefold().rstrip(".") == _NOT_FOUND.casefold().rstrip("."):
        result.refused = not orders
        answer = _NOT_FOUND if result.refused else ""

    for order in dict.fromkeys(orders):
        if order.upper() not in _order_ids(answer):
            answer = f"{answer} Заказ {order} создан.".strip()
    if unavailable and _ERP_UNAVAILABLE not in answer:
        answer = f"{answer} {_ERP_UNAVAILABLE}".strip()

    result.answer = answer
    return result
