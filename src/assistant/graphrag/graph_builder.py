"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен
    # raise NotImplementedError("Реализуйте build_graph")
    graph = nx.DiGraph()

    for result in results:
        chunk_id = result.chunk_id
        entities = list(result.entities)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            source_id = relation.subject.node_id
            target_id = relation.object.node_id
            graph.add_edge(source_id, target_id, type=relation.type)
            edge = graph.edges[source_id, target_id]
            edge.setdefault("sources", set()).add(chunk_id)

            entities.extend((relation.subject, relation.object))

        for entity in entities:
            node_id = entity.node_id
            graph.add_node(node_id, type=entity.type, name=entity.name)
            graph.nodes[node_id].setdefault("sources", set()).add(chunk_id)

    for _, node in graph.nodes(data=True):
        node["sources"] = sorted(node["sources"])

    for _, _, edge in graph.edges(data=True):
        edge["sources"] = sorted(edge["sources"])
        edge["weight"] = len(edge["sources"])

    conflicts = 0
    for source_id, target_id, edge in list(graph.edges(data=True)):
        if source_id >= target_id or edge["type"] != "REPLACED_BY":
            continue

        reverse = graph.get_edge_data(target_id, source_id)
        if reverse is None or reverse["type"] != "REPLACED_BY":
            continue

        conflicts += 1
        if edge["weight"] <= reverse["weight"]:
            graph.remove_edge(source_id, target_id)
        if reverse["weight"] <= edge["weight"]:
            graph.remove_edge(target_id, source_id)

    graph.graph["conflicts"] = conflicts
    return graph
