"""The shared material layer: one record per metal, per-field overrides."""

import numpy as np
import pytest

from laserttm.materials import (
    MATERIALS,
    k_table,
    resolve_material,
)


def test_conductivity_split_sums_to_total():
    """The two historical preset shapes carried the same numbers: the
    lattice-family total is the optical family's ke0 + kl."""
    for key, mat in MATERIALS.items():
        if mat.ke0 is None:
            continue
        assert mat.ke0 + mat.kl == pytest.approx(mat.k_total), key


def test_preset_values_are_unchanged():
    w = resolve_material({"material": "W"}, needs_optical=True)
    assert (w.gamma, w.cl, w.g_ep) == (137.3, 2.54e6, 1.65e17)
    assert (w.ke0, w.kl, w.alpha_opt) == (150.0, 24.0, 5.88e7)
    assert w.k_total == 174.0

    cu = resolve_material({"material": "Cu"}, needs_optical=False)
    assert (cu.gamma, cu.cl, cu.g_ep, cu.k_total) == (98.0, 3.45e6, 0.90e17, 401.0)


def test_material_lookup_is_case_insensitive():
    assert resolve_material({"material": "w"}, needs_optical=False).key == "w"
    assert resolve_material({"material": "AL"}, needs_optical=False).key == "al"


def test_melting_point_is_per_material():
    t_melt = {k: resolve_material({"material": k}, needs_optical=False).t_melt_c
              for k in ("w", "cu", "al", "au")}
    assert t_melt == {"w": 3422.0, "cu": 1085.0, "al": 660.0, "au": 1064.0}


def test_gold_is_available_to_the_lattice_family():
    au = resolve_material({"material": "Au"}, needs_optical=False)
    assert (au.gamma, au.cl, au.g_ep, au.k_total) == (67.0, 2.49e6, 1.40e16, 317.0)


def test_gold_is_rejected_where_optical_data_is_needed():
    with pytest.raises(ValueError, match="optical absorption"):
        resolve_material({"material": "Au"}, needs_optical=True)


def test_unknown_material_names_the_alternatives():
    with pytest.raises(ValueError, match="Unknown material"):
        resolve_material({"material": "unobtainium"}, needs_optical=False)


def test_custom_defaults_to_tungsten_in_both_families():
    lattice = resolve_material({"material": "custom"}, needs_optical=False)
    optical = resolve_material({"material": "custom"}, needs_optical=True)
    w = MATERIALS["w"]
    assert (lattice.gamma, lattice.cl, lattice.g_ep) == (w.gamma, w.cl, w.g_ep)
    assert lattice.k_total == w.k_total
    assert (optical.ke0, optical.kl, optical.alpha_opt) == (w.ke0, w.kl, w.alpha_opt)


def test_custom_reads_the_manual_fields():
    mat = resolve_material(
        {"material": "custom", "gamma_manual": 100.0, "Cl_manual": 3e6,
         "G_manual": 2e17, "kl_manual": 200.0},
        needs_optical=False)
    assert (mat.gamma, mat.cl, mat.g_ep, mat.k_total) == (100.0, 3e6, 2e17, 200.0)


def test_per_field_override_keeps_the_rest_of_the_preset():
    mat = resolve_material({"material": "W", "alpha_opt": 1.0 / 23e-9},
                           needs_optical=True)
    assert mat.alpha_opt == pytest.approx(1.0 / 23e-9)
    assert mat.delta_opt == pytest.approx(23e-9)
    # Everything else is still tungsten, including the measured k(T) table.
    assert (mat.gamma, mat.cl, mat.g_ep, mat.ke0, mat.kl) == (137.3, 2.54e6,
                                                              1.65e17, 150.0, 24.0)
    assert mat.measured_k_table


def test_delta_opt_is_the_reciprocal_of_alpha_opt():
    by_delta = resolve_material({"material": "W", "delta_opt": 23e-9},
                                needs_optical=True)
    by_alpha = resolve_material({"material": "W", "alpha_opt": 1.0 / 23e-9},
                                needs_optical=True)
    assert by_delta.alpha_opt == pytest.approx(by_alpha.alpha_opt)


def test_optical_overrides_are_rejected_by_the_lattice_family():
    with pytest.raises(ValueError, match="Leff"):
        resolve_material({"material": "W", "alpha_opt": 1e8}, needs_optical=False)
    with pytest.raises(ValueError, match="ke0"):
        resolve_material({"material": "W", "ke0": 100.0}, needs_optical=False)


def test_conductivity_override_recomputes_the_total():
    mat = resolve_material({"material": "W", "ke0": 100.0, "kl": 20.0},
                           needs_optical=True)
    assert (mat.ke0, mat.kl, mat.k_total) == (100.0, 20.0, 120.0)


def test_changed_conductivity_drops_the_measured_table():
    mat = resolve_material({"material": "W", "kl": 200.0}, needs_optical=False)
    assert mat.k_total == 200.0
    assert not mat.measured_k_table
    _, k_tab = k_table(mat)
    np.testing.assert_array_equal(k_tab, [200.0, 200.0])


def test_restating_a_preset_conductivity_is_a_no_op():
    """Passing tungsten's own k back in must not switch it off the measured
    k(T) curve: that would silently change computed temperatures."""
    plain = resolve_material({"material": "W"}, needs_optical=False)
    restated = resolve_material({"material": "W", "kl": 174.0},
                                needs_optical=False)
    assert restated.measured_k_table == plain.measured_k_table is True
    np.testing.assert_array_equal(k_table(restated)[1], k_table(plain)[1])

    # Same for the optical family, where kl is the lattice term.
    opt_plain = resolve_material({"material": "W"}, needs_optical=True)
    opt_restated = resolve_material({"material": "W", "ke0": 150.0, "kl": 24.0},
                                    needs_optical=True)
    assert opt_restated.measured_k_table
    np.testing.assert_array_equal(k_table(opt_restated)[1],
                                  k_table(opt_plain)[1])


def test_k_table_choice_can_be_forced():
    kept = resolve_material({"material": "W", "kl": 200.0, "kTable": "measured"},
                            needs_optical=False)
    assert kept.measured_k_table
    assert k_table(kept)[0].size > 2  # the measured tungsten curve

    flat = resolve_material({"material": "W", "kTable": "constant"},
                            needs_optical=False)
    assert not flat.measured_k_table
    np.testing.assert_array_equal(k_table(flat)[1], [174.0, 174.0])

    with pytest.raises(ValueError, match="kTable"):
        resolve_material({"material": "W", "kTable": "bogus"}, needs_optical=False)


def test_tungsten_gets_the_measured_table_and_others_a_flat_one():
    t_w, k_w = k_table(resolve_material({"material": "W"}, needs_optical=False))
    assert t_w.size > 2
    assert k_w.size == t_w.size

    _, k_cu = k_table(resolve_material({"material": "Cu"}, needs_optical=False))
    np.testing.assert_array_equal(k_cu, [401.0, 401.0])


def test_constant_only_forces_a_flat_table_for_tungsten():
    mat = resolve_material({"material": "W"}, needs_optical=True)
    _, k_flat = k_table(mat, constant_only=True)
    np.testing.assert_array_equal(k_flat, [174.0, 174.0])


def test_overrides_can_come_from_a_separate_mapping():
    """scanning_beam merges a defaults table into its params, so it passes the
    caller's untouched dict as the override source."""
    cfg = {"material": "Cu", "gamma": 137.3}       # 'gamma' is a merged default
    mat = resolve_material(cfg, needs_optical=False, overrides={})
    assert mat.gamma == 98.0                       # copper's own value, not the default
