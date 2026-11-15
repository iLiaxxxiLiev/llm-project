"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # TODO: опорные сущности, отбор фактов, запрос к модели, источники
    # raise NotImplementedError("Реализуйте local_search")
    from itertools import combinations, pairwise
    import networkx as nx

    from assistant.infra.grounding import code_type, find_codes, find_names

    graph = kb.graph
    not_found = "В базе знаний не найдено."

    anchors = {f"{code_type(code)}:{code}" for code in find_codes(question)}
    if any(node not in graph for node in anchors):
        return Answer(text=not_found, refused=True)

    for entity_type, name in find_names(question, kb.vocabulary):
        node = f"{entity_type}:{name}"
        if node in graph:
            anchors.add(node)

    if not anchors:
        return Answer(text=not_found, refused=True)

    def find_edges() -> set[tuple[str, str]]:
        """Отбирает направленные рёбра по опорным узлам."""
        undirected = graph.to_undirected(as_view=True)

        if len(anchors) == 1:
            anchor = next(iter(anchors))
            nearby = nx.single_source_shortest_path_length(undirected, anchor, cutoff=2)
            return set(graph.subgraph(nearby).edges())

        edges: set[tuple[str, str]] = set()
        path_nodes: set[str] = set()

        for start, end in combinations(sorted(anchors), 2):
            for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
                path_nodes.update(path)
                for left, right in pairwise(path):
                    if graph.has_edge(left, right):
                        edges.add((left, right))
                    if graph.has_edge(right, left):
                        edges.add((right, left))

        pending = [node for node in path_nodes if graph.nodes[node]["type"] == "Part"]
        seen = set(pending)
        while pending:
            source = pending.pop()
            for target, edge in graph[source].items():
                if edge["type"] != "REPLACED_BY":
                    continue

                edges.add((source, target))
                if target not in seen:
                    seen.add(target)
                    pending.append(target)

        return edges

    facts: list[str] = []
    numbered_facts: list[str] = []
    sources: set[str] = set()
    context_length = 0

    for source, target in sorted(find_edges()):
        edge = graph.edges[source, target]
        source_name = graph.nodes[source]["name"]
        target_name = graph.nodes[target]["name"]
        fact = f"{source_name} -{edge['type']}-> {target_name}"
        line = f"{len(numbered_facts) + 1}. {fact}"
        added_length = len(line) + (1 if numbered_facts else 0)

        if context_length + added_length > 8000:
            break

        facts.append(fact)
        numbered_facts.append(line)
        context_length += added_length
        sources.update(chunk_id.partition("#")[0] for chunk_id in edge["sources"])

    if not facts:
        return Answer(text=not_found, refused=True)

    fact_context = "\n".join(numbered_facts)
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Ответь кратко и только по приведённым фактам. "
                    "Используй факты о сущностях из вопроса. "
                    "REPLACED_BY направлена от старой детали к её замене. "
                    "Если вопрос касается актуальной детали, проследи цепочку замен "
                    "и назови код её последней детали. "
                    f"Если фактов недостаточно, напиши ровно «{not_found}»."
                ),
            },
            {"role": "user", "content": f"Вопрос\n{question}\n\nФакты из графа\n{fact_context}"},
        ]
    )
    text = response.content.strip()
    if not text or text.casefold().rstrip(".") == not_found.casefold().rstrip("."):
        return Answer(text=not_found, refused=True)

    return Answer(text=text, sources=sorted(sources), facts=facts)



def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    # raise NotImplementedError("Реализуйте global_search")
    import json

    from assistant.infra.grounding import code_type, find_codes
    from assistant.infra.schemas import MAP_SCHEMA

    not_found = "В базе знаний не найдено."

    if any(f"{code_type(code)}:{code}" not in kb.graph for code in find_codes(question)):
        return Answer(text=not_found, refused=True)

    partial_answers: list[str] = []
    sources: set[str] = set()

    for community in kb.communities:
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Дай частичный ответ на вопрос по одной сводке сообщества. "
                        "Сохраняй названия линеек, модели, коды деталей и числа, полезные для ответа. "
                        "Дай оценку полезности от 0 до 10. "
                        "Если в сводке нет подходящих сведений, верни пустой ответ с оценкой 0."
                    ),
                },
                {"role": "user", "content": f"Вопрос\n{question}\n\nСводка сообщества\n{community.summary}"},
            ],
            json_schema=MAP_SCHEMA,
        )

        try:
            partial = json.loads(response.content)
        except (ValueError, RecursionError):
            continue

        if not isinstance(partial, dict):
            continue

        score = partial.get("score")
        answer = partial.get("answer")
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 10:
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue

        partial_answers.append(answer.strip())
        for node in community.nodes:
            sources.update(chunk_id.partition("#")[0] for chunk_id in kb.graph.nodes[node]["sources"])

    if not partial_answers:
        return Answer(text=not_found, refused=True)

    context = "\n".join(f"{index}. {answer}" for index, answer in enumerate(partial_answers, start=1))
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы в один ответ на вопрос. "
                    "Используй только переданные сведения. "
                    "Для сравнения частот учитывай сопоставимые подсчёты из разных сообществ. "
                    f"Если сведений недостаточно, напиши ровно «{not_found}»."
                ),
            },
            {"role": "user", "content": f"Вопрос\n{question}\n\nЧастичные ответы\n{context}"},
        ]
    )
    text = response.content.strip()
    if not text or text.casefold().rstrip(".") == not_found.casefold().rstrip("."):
        return Answer(text=not_found, refused=True)

    return Answer(text=text, sources=sorted(sources))
