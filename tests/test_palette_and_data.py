from dacolor.data import load_schemes
from dacolor.palette import build_palette, build_source_id_map, load_source_palette_rgb


def test_authoritative_palette_is_deduplicated_and_traceable():
    source = load_source_palette_rgb()
    palette = build_palette()
    source_id_map = build_source_id_map()

    assert len(source) == 162
    assert len(palette) == 151
    assert len({tuple(item["rgb"]) for item in palette}) == 151
    assert source_id_map[0] == source_id_map[9]
    assert source_id_map[130] == source_id_map[139]
    assert palette[source_id_map[80]]["rgb"] == [0, 0, 0]
    assert palette[source_id_map[132]]["rgb"] == [227, 26, 28]
    assert all(source_id in palette[canonical_id]["source_ids"] for source_id, canonical_id in source_id_map.items())


def test_all_dataset_color_ids_map_to_the_unique_palette():
    palette = build_palette()
    schemes = load_schemes("data/data.txt", color_id_map=build_source_id_map())
    used = {color_id for scheme in schemes for color_id in scheme.color_ids}

    assert len(schemes) == 25040
    assert used == set(range(len(palette)))
    assert all(len(scheme.color_ids) == len(set(scheme.color_ids)) for scheme in schemes)
