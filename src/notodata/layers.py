"""Registry of every GSI imagery layer considered for the build (Phase 5a).

Ids are copied exactly from the GSI tile list, including the four 5 January
ids that carry an extra underscore after `20240102`. The baseline URL template
comes from `notodata.config`, never from a literal here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from notodata.config import CONFIG

TILE_LIST_URL = "https://maps.gsi.go.jp/development/ichiran.html"
_XYZ = "https://cyberjapandata.gsi.go.jp/xyz/{layer_id}/{{z}}/{{x}}/{{y}}.{ext}"
_FOUND_IN = "GSI tile list page (ichiran.html)"


@dataclass(frozen=True)
class Layer:
    layer_id: str
    role_candidate: str
    """One of `baseline`, `pre` or `post`."""
    capture_dates: tuple[date, ...]
    """One date, or the first and last day of a range; empty when unknown."""
    capture_dates_note: str
    districts: tuple[str, ...]
    url_template: str
    ext: str
    source_url: str
    found_in: str


def _post(layer_id: str, dates: tuple[date, ...], district: str) -> Layer:
    return Layer(
        layer_id=layer_id,
        role_candidate="post",
        capture_dates=dates,
        capture_dates_note="",
        districts=(district,),
        url_template=_XYZ.format(layer_id=layer_id, ext="png"),
        ext="png",
        source_url=TILE_LIST_URL,
        found_in=_FOUND_IN,
    )


def _d(day: int) -> tuple[date, ...]:
    return (date(2024, 1, day),)


_BASELINE = Layer(
    layer_id=CONFIG.baseline_imagery.layer_id,
    role_candidate="baseline",
    capture_dates=(),
    capture_dates_note="composite; per-pixel date unknown",
    districts=("All Japan",),
    url_template=CONFIG.baseline_imagery.url_template,
    ext="jpg",
    source_url=TILE_LIST_URL,
    found_in=_FOUND_IN,
)

_PRE = tuple(
    Layer(
        layer_id=f"nendophoto{year}",
        role_candidate="pre",
        capture_dates=(),
        capture_dates_note=f"fiscal year {year}",
        districts=("Varies by year",),
        url_template=_XYZ.format(layer_id=f"nendophoto{year}", ext="png"),
        ext="png",
        source_url=TILE_LIST_URL,
        found_in=_FOUND_IN,
    )
    for year in range(2018, 2023)
)

_POST = (
    _post(
        "20240102noto_0405_0426do",
        (date(2024, 4, 5), date(2024, 4, 26)),
        "Whole Noto peninsula",
    ),
    _post("20240102noto_wazimanishi_0117do", _d(17), "Wajima West"),
    _post("20240102noto_anamizu_0117do", _d(17), "Anamizu"),
    _post("20240102noto_nanao_0117do", _d(17), "Nanao"),
    _post("20240102noto_suzu_0114do", _d(14), "Suzu"),
    _post("20240102noto_wazimahigashi_0114do", _d(14), "Wajima East"),
    _post("20240102noto_anamizu_0114do", _d(14), "Anamizu"),
    _post("20240102noto_wazimanaka_0111do", _d(11), "Wajima Central"),
    _post("20240102noto_wazimanishi_0111do", _d(11), "Wajima West"),
    _post("20240102noto_anamizu_0111do", _d(11), "Anamizu"),
    _post("20240102_noto_suzu_0105do", _d(5), "Suzu"),
    _post("20240102_noto_wazimanaka_0105do", _d(5), "Wajima Central"),
    _post("20240102_noto_anamizu_0105do", _d(5), "Anamizu"),
    _post("20240102_noto_nanao_0105do", _d(5), "Nanao"),
    _post("20240102noto_suzu_0102do", _d(2), "Suzu"),
    _post("20240102noto_wazimahigashi_0102do", _d(2), "Wajima East"),
    _post("20240102noto_wazimanaka_0102do", _d(2), "Wajima Central"),
)

LAYERS: tuple[Layer, ...] = (_BASELINE, *_PRE, *_POST)


def get_layer(layer_id: str) -> Layer:
    for layer in LAYERS:
        if layer.layer_id == layer_id:
            return layer
    raise KeyError(layer_id)


def by_role(role: str) -> tuple[Layer, ...]:
    return tuple(layer for layer in LAYERS if layer.role_candidate == role)
