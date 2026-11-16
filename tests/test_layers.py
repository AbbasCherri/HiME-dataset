import dataclasses
from datetime import date

import pytest

from notodata import layers
from notodata.config import CONFIG

JAN_UNDERSCORE = {
    "20240102_noto_suzu_0105do",
    "20240102_noto_wazimanaka_0105do",
    "20240102_noto_anamizu_0105do",
    "20240102_noto_nanao_0105do",
}


def test_layer_registry_complete():
    reg = layers.LAYERS
    ids = [layer.layer_id for layer in reg]
    assert len(reg) == 23
    assert len(set(ids)) == 23
    assert set(ids).issuperset(JAN_UNDERSCORE)
    assert "20240102noto_suzu_0105do" not in ids
    roles = [layer.role_candidate for layer in reg]
    assert roles.count("baseline") == 1
    assert roles.count("pre") == 5
    assert roles.count("post") == 17
    assert ids[0] == CONFIG.baseline_imagery.layer_id
    nanao = layers.get_layer("20240102noto_nanao_0117do")
    assert nanao.capture_dates == (date(2024, 1, 17),)
    assert nanao.districts == ("Nanao",)
    april = layers.get_layer("20240102noto_0405_0426do")
    assert april.capture_dates == (date(2024, 4, 5), date(2024, 4, 26))
    for year in range(2018, 2023):
        pre = layers.get_layer(f"nendophoto{year}")
        assert pre.role_candidate == "pre"
        assert pre.capture_dates_note == f"fiscal year {year}"


def test_layer_urls_and_baseline():
    base = layers.get_layer(CONFIG.baseline_imagery.layer_id)
    assert base.url_template == CONFIG.baseline_imagery.url_template
    assert base.ext == "jpg"
    assert "composite" in base.capture_dates_note
    for layer in layers.LAYERS:
        assert layer.url_template.endswith("{z}/{x}/{y}." + layer.ext)
        assert f"/xyz/{layer.layer_id}/" in layer.url_template
        assert layer.source_url
        assert layer.found_in


def test_layer_frozen_and_lookup():
    layer = layers.LAYERS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        layer.layer_id = "x"  # type: ignore[misc]
    with pytest.raises(KeyError):
        layers.get_layer("nope")
    assert [x.layer_id for x in layers.by_role("pre")] == [
        f"nendophoto{y}" for y in range(2018, 2023)
    ]


def test_full_urls():
    post = layers.get_layer("20240102_noto_nanao_0105do")
    assert post.url_template == (
        "https://cyberjapandata.gsi.go.jp/xyz/20240102_noto_nanao_0105do/{z}/{x}/{y}.png"
    )
    pre = layers.get_layer("nendophoto2020")
    assert pre.url_template == (
        "https://cyberjapandata.gsi.go.jp/xyz/nendophoto2020/{z}/{x}/{y}.png"
    )
    assert pre.ext == "png"


POST_TABLE = [
    ("20240102noto_0405_0426do", (date(2024, 4, 5), date(2024, 4, 26)), "Whole Noto peninsula"),
    ("20240102noto_wazimanishi_0117do", (date(2024, 1, 17),), "Wajima West"),
    ("20240102noto_anamizu_0117do", (date(2024, 1, 17),), "Anamizu"),
    ("20240102noto_nanao_0117do", (date(2024, 1, 17),), "Nanao"),
    ("20240102noto_suzu_0114do", (date(2024, 1, 14),), "Suzu"),
    ("20240102noto_wazimahigashi_0114do", (date(2024, 1, 14),), "Wajima East"),
    ("20240102noto_anamizu_0114do", (date(2024, 1, 14),), "Anamizu"),
    ("20240102noto_wazimanaka_0111do", (date(2024, 1, 11),), "Wajima Central"),
    ("20240102noto_wazimanishi_0111do", (date(2024, 1, 11),), "Wajima West"),
    ("20240102noto_anamizu_0111do", (date(2024, 1, 11),), "Anamizu"),
    ("20240102_noto_suzu_0105do", (date(2024, 1, 5),), "Suzu"),
    ("20240102_noto_wazimanaka_0105do", (date(2024, 1, 5),), "Wajima Central"),
    ("20240102_noto_anamizu_0105do", (date(2024, 1, 5),), "Anamizu"),
    ("20240102_noto_nanao_0105do", (date(2024, 1, 5),), "Nanao"),
    ("20240102noto_suzu_0102do", (date(2024, 1, 2),), "Suzu"),
    ("20240102noto_wazimahigashi_0102do", (date(2024, 1, 2),), "Wajima East"),
    ("20240102noto_wazimanaka_0102do", (date(2024, 1, 2),), "Wajima Central"),
]


@pytest.mark.parametrize(("layer_id", "dates", "district"), POST_TABLE)
def test_post_layer_table(layer_id, dates, district):
    layer = layers.get_layer(layer_id)
    assert layer.role_candidate == "post"
    assert layer.capture_dates == dates
    assert layer.districts == (district,)


def test_post_table_covers_all_post_layers():
    assert {row[0] for row in POST_TABLE} == {x.layer_id for x in layers.by_role("post")}
