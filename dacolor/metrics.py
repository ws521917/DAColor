from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RankingMetrics:
    count: int = 0
    acc1: float = 0.0
    acc5: float = 0.0
    acc10: float = 0.0
    mrr: float = 0.0

    def update(self, rank: int) -> None:
        self.count += 1
        self.acc1 += float(rank <= 1)
        self.acc5 += float(rank <= 5)
        self.acc10 += float(rank <= 10)
        self.mrr += 1.0 / rank

    def as_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0, "acc@1": 0.0, "acc@5": 0.0, "acc@10": 0.0, "mrr": 0.0}
        return {
            "count": self.count,
            "acc@1": self.acc1 / self.count,
            "acc@5": self.acc5 / self.count,
            "acc@10": self.acc10 / self.count,
            "mrr": self.mrr / self.count,
        }


@dataclass
class QuerySizeMetrics:
    overall: RankingMetrics = field(default_factory=RankingMetrics)
    by_query_size: dict[int, RankingMetrics] = field(
        default_factory=lambda: {1: RankingMetrics(), 2: RankingMetrics(), 3: RankingMetrics()}
    )

    def update(self, query_size: int, rank: int) -> None:
        self.overall.update(rank)
        self.by_query_size[int(query_size)].update(rank)

    def as_dict(self) -> dict[str, dict[str, float]]:
        payload = {"overall": self.overall.as_dict()}
        for query_size, metric in self.by_query_size.items():
            payload[f"query_size_{query_size}"] = metric.as_dict()
        return payload
