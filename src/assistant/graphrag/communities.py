"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    detect_communities строит временный неориентированный граф и запускает на нём алгоритм Лувена.
    Алгоритм учитывает веса связей и ищет группы, внутри которых узлы связаны плотнее.
    Если между двумя узлами есть рёбра в обе стороны, складываем их веса. Фиксированный seed
    и сортировка делают результат воспроизводимым для одного и того же входного графа.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    # raise NotImplementedError("Реализуйте detect_communities")
    if graph.number_of_edges() == 0:
        return [
            Community(community_id=index, nodes=[node])
            for index, node in enumerate(sorted(graph.nodes))
        ]

    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))

    for source, target, edge in sorted(graph.edges(data=True)):
        previous_weight = undirected.get_edge_data(source, target, default={}).get("weight", 0)
        undirected.add_edge(source, target, weight=previous_weight + edge.get("weight", 1))

    groups = nx.community.louvain_communities(undirected, weight="weight", seed=42)
    ordered_groups = sorted(sorted(group) for group in groups)
    return [
        Community(community_id=index, nodes=nodes)
        for index, nodes in enumerate(ordered_groups)
    ]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    summarize_community берёт исходящие связи узлов группы из исходного направленного графа.
    Так связь сохраняется в сводке своей группы, даже если её второй узел попал в другую.
    Для рёбер FAILED код суммирует веса по моделям и по узлам внутри линеек.
    Если две модели «Арктики» связаны с пусковым реле рёбрами FAILED с весами 37 и 2,
    модель получит готовую сумму 39. Затем она одним вызовом формулирует сводку,
    а Python ограничивает результат 150 словами.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    # raise NotImplementedError("Реализуйте summarize_community")
    facts: list[str] = []
    failures_by_model: dict[str, int] = {}
    failures_by_line_and_component: dict[tuple[str, str], int] = {}

    for source, target, edge in sorted(graph.out_edges(community.nodes, data=True)):
        relation_type = edge["type"]
        weight = edge.get("weight", 1)
        facts.append(f"{source} -{relation_type}-> {target} (вес {weight})")

        if relation_type != "FAILED":
            continue

        failures_by_model[source] = failures_by_model.get(source, 0) + weight
        for product_line, membership in graph[source].items():
            if membership["type"] == "BELONGS_TO":
                key = (product_line, target)
                failures_by_line_and_component[key] = failures_by_line_and_component.get(key, 0) + weight

    model_stats = [
        f"{model} — {weight}"
        for model, weight in sorted(failures_by_model.items(), key=lambda item: (-item[1], item[0]))
    ]
    component_stats = [
        f"{line}, {component} — {weight}"
        for (line, component), weight in sorted(
            failures_by_line_and_component.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    context = "\n".join(
        [
            "Узлы сообщества",
            ", ".join(sorted(community.nodes)),
            "Отказы по моделям (суммы весов FAILED)",
            *(model_stats or ["Нет данных"]),
            "Отказы по узлам в каждой линейке (суммы весов FAILED)",
            *(component_stats or ["Нет данных"]),
            "Связи узлов сообщества",
            *(facts or ["Нет связей"]),
        ]
    )
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши сводку сообщества графа на русском языке, до 150 слов. "
                    "Укажи основные сущности и связи, частые отказы и замены деталей, если они есть. "
                    "Используй только переданные факты и готовые подсчёты. "
                    "Сохраняй точные коды, названия и числа. "
                    "Относи статистику отказов к этому сообществу."
                ),
            },
            {"role": "user", "content": context},
        ]
    )
    return " ".join(response.content.split()[:150])
