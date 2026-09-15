"""SHOWCASE (not part of the competition submission) -- one long table, N
SO-101 arms along its back edge, objects passed down the line (2026-09-15).

Same pinned Menagerie arm, same actuators, same contact physics as the
submission; the arms' motion is scripted choreography (see
record_relay_showcase.py), not the planner.
"""
from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from omni_q.intel_sim import (  # noqa: E402
    ARM_ASSETS, ARM_XML, PAD_FRICTION, PAD_SOLIMP, PAD_SOLREF, _free_body, _hollow_cup, _lipped_plate, _prefixed,
)

ARM_Y = 0.20          # bases along the back edge
SPACING = 0.50        # between arms; SO-101 reaches ~0.33 m, so neighbours share a 0.16 m band
TABLE_HALF_Y = 0.32


def arm_x(i: int, n: int) -> float:
    return (i - (n - 1) / 2.0) * SPACING


def relay_xml(n_arms: int = 6, blocks: int = 2) -> str:
    source = ET.parse(ARM_XML).getroot()
    root = ET.Element("mujoco", {"model": "omni_q_relay_showcase"})
    for tag in ("compiler", "option", "asset", "default"):
        node = source.find(tag)
        if node is not None:
            root.append(copy.deepcopy(node))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(visual, "headlight", {"ambient": ".30 .30 .30", "diffuse": ".35 .35 .35", "specular": ".1 .1 .1"})
    asset = root.find("asset")
    ET.SubElement(asset, "texture", {"name": "room_sky", "type": "skybox", "builtin": "gradient",
                                     "rgb1": ".55 .62 .70", "rgb2": ".18 .20 .24", "width": "256", "height": "256"})
    ET.SubElement(asset, "texture", {"name": "room_floor_tex", "type": "2d", "builtin": "checker",
                                     "rgb1": ".32 .30 .28", "rgb2": ".26 .24 .22", "width": "256", "height": "256"})
    ET.SubElement(asset, "material", {"name": "room_floor", "texture": "room_floor_tex", "texrepeat": "6 6",
                                      "texuniform": "true", "reflectance": ".05"})
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"name": "key", "pos": "0 -1 3", "dir": "0 0.3 -1", "directional": "true", "diffuse": ".8 .8 .8"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "pos": "0 0 -.75", "size": "0 0 .05", "material": "room_floor"})
    half_x = (n_arms - 1) / 2.0 * SPACING + 0.45
    ET.SubElement(wb, "geom", {"name": "table", "type": "box", "pos": f"0 -0.06 -0.05", "size": f"{half_x:.3f} {TABLE_HALF_Y} .05",
                               "rgba": ".23 .14 .08 1", "friction": "1 .005 .0001"})
    for i, lx in enumerate((-half_x + 0.05, half_x - 0.05, 0.0)):
        for ly in (-TABLE_HALF_Y - 0.06 + 0.03, TABLE_HALF_Y - 0.06 - 0.03):
            ET.SubElement(wb, "geom", {"name": f"leg_{i}_{ly > 0}", "type": "box", "pos": f"{lx:.3f} {ly:.3f} -.425",
                                       "size": ".025 .025 .325", "rgba": ".20 .12 .07 1"})
    base = source.find("./worldbody/body[@name='Base']")
    source_actuators = source.find("actuator")
    source_excludes = source.findall("./contact/exclude")
    actuators = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    for i in range(n_arms):
        prefix = f"a{i + 1}"
        arm_body = _prefixed(base, prefix)
        arm_body.set("pos", f"{arm_x(i, n_arms):.3f} {ARM_Y} 0")
        for geom in arm_body.iter("geom"):
            if "jaw_pad_" in geom.attrib.get("name", ""):
                geom.set("friction", PAD_FRICTION); geom.set("solref", PAD_SOLREF); geom.set("solimp", PAD_SOLIMP)
        wb.append(arm_body)
        for exclude in source_excludes:
            ET.SubElement(contact, "exclude", {"body1": f"{prefix}_{exclude.attrib['body1']}", "body2": f"{prefix}_{exclude.attrib['body2']}"})
        for actuator in source_actuators:
            copied = copy.deepcopy(actuator)
            copied.set("name", f"{prefix}_{actuator.attrib['name']}")
            copied.set("joint", f"{prefix}_{actuator.attrib['joint']}")
            actuators.append(copied)
    # the things passed down the line: 4 cm blocks at the left end, and a plate at the right end to receive them
    colours = (".90 .35 .30 1", ".30 .60 .90 1", ".95 .80 .25 1", ".40 .80 .45 1")
    # the vertical-finger pinch works ~0.34-0.36 m from a base (measured on the
    # hand-off band); both spawns sit at that radius from arm 1's base
    spawns = [(-0.22, -0.06), (-0.10, -0.14), (-0.30, 0.02), (-0.18, -0.20)]
    for b in range(blocks):
        sx, sy = spawns[b % len(spawns)]
        body = _free_body(f"block_{b + 1}", f"{arm_x(0, n_arms) + sx:.3f} {sy:.3f} .02")
        ET.SubElement(body, "geom", {"name": f"block_{b + 1}", "type": "box", "size": ".02 .02 .02", "mass": ".05",
                                     "friction": "1.2 .006 .0002", "rgba": colours[b % len(colours)]})
        wb.append(body)
    wb.append(_lipped_plate("plate_end", f"{arm_x(n_arms - 1, n_arms) + 0.22:.3f} -0.06 .0105", rgba=".92 .92 .90 1",
                            mass=".20", friction="1.20 .006 .0002"))
    wb.append(_hollow_cup("cup_end", f"{arm_x(n_arms - 1, n_arms) + 0.30:.3f} 0.14 .045", rgba=".22 .58 .78 1",
                          mass=".10", friction="3.00 .020 .001", contact={}))
    ET.SubElement(wb, "camera", {"name": "line_overhead", "pos": "0 -0.06 3.2", "euler": "0 0 0", "fovy": "60"})
    return ET.tostring(root, encoding="unicode")


def load_relay_model(n_arms: int = 6, blocks: int = 2):
    import mujoco
    assets = {f"assets/{p.name}": p.read_bytes() for p in ARM_ASSETS.glob("*.stl")}
    return mujoco.MjModel.from_xml_string(relay_xml(n_arms, blocks), assets=assets)
