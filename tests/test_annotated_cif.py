import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from biotite.sequence import ProteinSequence
from biotite.structure.io.pdbx import CIFCategory, CIFFile

from protenix.data.constants import ATOM37_ORDER
from protenix.data.inference.json_maker import annotated_cif_to_input_json, ccd
from protenix.data.template.template_utils import TemplateHitFeaturizer


def test_annotated_cif_resolves_chain_files_and_preserves_disulfide(
    tmp_path, monkeypatch
):
    # This fixture contains only standard amino acids. Keep the unit test independent of a model installation.
    monkeypatch.setattr(ccd, "get_one_letter_code", ProteinSequence.convert_letter_3to1)
    monkeypatch.setattr(ccd, "get_mol_type", lambda _: "protein")
    cif = CIFFile.read(Path(__file__).resolve().parents[1] / "examples/2lwu.cif")
    source = tmp_path / "fixed.cif"
    cif.write(source)
    with pytest.raises(ValueError, match="cyclic"):
        annotated_cif_to_input_json(source)

    # Keep the deposited disulfide, leaving an open backbone.
    links = cif.block["struct_conn"]
    keep = links["conn_type_id"].as_array() == "disulf"
    cif.block["struct_conn"] = CIFCategory(
        {key: col.as_array()[keep] for key, col in links.items()}
    )
    cif.block["template_hits_by_chain_id"] = CIFCategory({"A": ["hits.hhr"]})
    cif.block["msa_paths_by_chain_id"] = CIFCategory({"A": ["query.a3m.gz"]})
    (tmp_path / "hits.hhr").touch()
    with gzip.open(tmp_path / "query.a3m.gz", "wt") as out:
        out.write(">query\nGLC\n")
    cif.write(source)
    job = annotated_cif_to_input_json(source)
    chain = job["sequences"][0]["proteinChain"]
    assert chain["id"] == ["A"]
    assert chain["templatesPath"] == str(tmp_path / "hits.hhr")
    assert chain["unpairedMsa"] == ">query\nGLC\n"
    (bond,) = job["covalent_bonds"]
    assert {bond["position1"], bond["position2"]} == {3, 12}
    assert bond["atom1"] == bond["atom2"] == "SG"

    cif.block["template_hits_by_chain_id"] = CIFCategory({"B": ["hits.hhr"]})
    cif.write(source)
    with pytest.raises(ValueError, match="chain B"):
        annotated_cif_to_input_json(source)


def test_whole_chain_uses_the_archived_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(ccd, "get_one_letter_code", ProteinSequence.convert_letter_3to1)
    monkeypatch.setattr(ccd, "get_mol_type", lambda _: "protein")
    reference = Path(__file__).resolve().parents[1] / "scripts/msa/data/mmcif/102m.cif"
    mirror = tmp_path / "mirror"
    (mirror / "02").mkdir(parents=True)
    with gzip.open(mirror / "02/102m.cif.gz", "wb") as out:
        out.write(reference.read_bytes())
    monkeypatch.setenv("PDB_MIRROR_PATH", str(mirror))
    cif = CIFFile.read(reference)
    sites = cif.block["atom_site"]
    keep = (sites["label_asym_id"].as_array() == "A") & np.isin(
        sites["label_alt_id"].as_array(), [".", "A"]
    )
    sites = CIFCategory({k: v.as_array()[keep] for k, v in sites.items()})
    cif.block["atom_site"] = sites
    if "struct_conn" in cif.block:
        del cif.block["struct_conn"]
    order = np.arange(sites.row_count)[::-1]
    coords = {"idx0": order}
    for i, axis in enumerate("xyz"):
        coords[f"val{i}"] = sites[f"Cartn_{axis}"].as_array(float)[order]
    category = "condition_coordinate_1_atom"
    cif.block[category] = CIFCategory(coords)
    query = tmp_path / "input.cif"
    cif.write(query)
    job = annotated_cif_to_input_json(query, template_dir=tmp_path)
    protein = job["sequences"][0]["proteinChain"]
    (direct,) = json.loads(Path(protein["templatesPath"]).read_text())
    archive_sites = CIFFile.deserialize(direct["mmcif"]).block["atom_site"]
    for key in sites:
        np.testing.assert_array_equal(
            archive_sites[key].as_array(), sites[key].as_array()
        )
    assert (
        direct["queryIndices"]
        == direct["templateIndices"]
        == list(range(len(protein["sequence"])))
    )
    reader = TemplateHitFeaturizer(str(mirror), _zero_center_positions=False)
    (features,) = reader.parse_json_templates([direct], protein["sequence"]).features
    assert features["template_all_atom_masks"].sum() == sites.row_count
    ca = sites["label_atom_id"].as_array() == "CA"
    expected = np.column_stack(
        [sites[f"Cartn_{axis}"].as_array(float)[ca] for axis in "xyz"]
    )
    np.testing.assert_allclose(
        features["template_all_atom_positions"][:, ATOM37_ORDER["CA"]],
        expected,
        atol=0.001,
    )

    # Folds of gapped, fixed structures must not condition unselected query residues.
    selected = [1, 3, 5]
    residue_ids = sites["label_seq_id"].as_array(int)[order] - 1
    keep = np.isin(residue_ids, selected)
    cif.block[category] = CIFCategory({k: v[keep] for k, v in coords.items()})
    cif.write(query)
    cropped = annotated_cif_to_input_json(query, template_dir=tmp_path)["sequences"][0][
        "proteinChain"
    ]
    (direct,) = json.loads(Path(cropped["templatesPath"]).read_text())
    assert direct["queryIndices"] == selected
    (features,) = reader.parse_json_templates([direct], cropped["sequence"]).features
    ca = ATOM37_ORDER["CA"]
    np.testing.assert_array_equal(
        np.flatnonzero(features["template_all_atom_masks"][:, ca]), selected
    )
    np.testing.assert_allclose(
        features["template_all_atom_positions"][selected, ca],
        expected[selected],
        atol=0.001,
    )
    cif.block[category] = CIFCategory(coords)
    cif.block["template_hits_by_chain_id"] = CIFCategory({"A": ["hits.hhr"]})
    (tmp_path / "hits.hhr").touch()
    cif.write(query)
    with pytest.raises(
        ValueError, match="Both a deposited self-template and a template file"
    ):
        annotated_cif_to_input_json(query, template_dir=tmp_path)
    del cif.block["template_hits_by_chain_id"]

    coords["val0"][0] += 2
    cif.block[category] = CIFCategory(coords)
    cif.write(query)
    with pytest.raises(ValueError, match="differ from the deposited chain"):
        annotated_cif_to_input_json(query, template_dir=tmp_path)
    cif.block[category] = CIFCategory({k: v[1:] for k, v in coords.items()})
    cif.write(query)
    with pytest.raises(ValueError, match="every atom"):
        annotated_cif_to_input_json(query, template_dir=tmp_path)
