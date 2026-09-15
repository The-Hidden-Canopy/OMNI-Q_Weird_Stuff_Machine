"""SHOWCASE (not part of the competition submission) -- a fleet scene built
around the real dual-arm table (2026-09-15).

Takes the submission's own generated MJCF (``dual_so101_xml``) and extends
it, additively, with:

* extra "units": more tables, each with its own pair of SO-101 arms (same
  pinned Menagerie asset, same actuators), cups and plates to work on;
* ``drone_01``: the Menagerie Skydio X2 model as a mocap (kinematic) body;
* ``mobile_01``: a small wheeled rover, mocap-driven, carrying a tray with a
  cup on it (the cup is a real free body riding on the tray).

What is real and what is scripted, so nobody is misled by the footage:

* Unit 1 (the two centre arms) is the actual submission stack -- OMNI plans,
  the engine sequences, both arms execute under the same physics and
  verifiers as every other clip in the folder.
* The extra units' arms are physically simulated (same servos, same contact
  physics) but their motion is scripted choreography, not the planner.
* The drone and the rover are kinematic showcase actors: they move along
  scripted paths; nothing is flown or driven by a controller.

The submission code is not modified: the extension happens on the XML
string in this process only, by wrapping ``load_dual_so101_model``.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from omni_q import intel_sim  # noqa: E402
from omni_q.intel_sim import (  # noqa: E402
    ARM_ASSETS, ARM_XML, PAD_FRICTION, PAD_SOLIMP, PAD_SOLREF, _hollow_cup, _lipped_plate, _prefixed,
)

MENAGERIE = Path.home() / "AppData" / "Local" / "Temp" / "claude" / \
    "C--Users-damio-OneDrive-Documents-GitHub-IDA-TRAIN-V2" / "4d0ed848-645b-4064-8e50-825fa786fd35" / "scratchpad" / "menagerie"
SKYDIO = MENAGERIE / "skydio_x2" / "assets"

# unit index -> table centre x (the submission table is unit 1 at x = 0, y = -0.10)
UNIT_X = {2: 1.15, 3: -1.15, 4: 2.30}
TABLE_Y = -0.10
DRONE_PAD = (-1.15, 1.10, -0.75 + 0.02)
ROVER_START = (3.6, 1.2, -0.75)


def unit_arm_names(unit: int) -> tuple[str, str]:
    return f"u{unit}_left", f"u{unit}_right"


def extend_fleet_xml(xml: str, *, units: int = 4, drone: bool = True, rover: bool = True) -> str:
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    actuators = root.find("actuator")
    contact = root.find("contact")
    asset = root.find("asset")
    source = ET.parse(ARM_XML).getroot()
    base = source.find("./worldbody/body[@name='Base']")
    source_actuators = source.find("actuator")
    source_excludes = source.findall("./contact/exclude")

    # -- extra units: table, legs, arm pair, a cup and a plate each ---------
    for unit in range(2, units + 1):
        ux = UNIT_X[unit]
        ET.SubElement(worldbody, "geom", {
            "name": f"u{unit}_table", "type": "box", "pos": f"{ux} {TABLE_Y} -0.05", "size": ".42 .36 .05",
            "rgba": ".23 .14 .08 1", "friction": "1 .005 .0001",
        })
        for lx, ly in ((-.38, -.42), (.38, -.42), (-.38, .22), (.38, .22)):
            ET.SubElement(worldbody, "geom", {
                "name": f"u{unit}_leg_{lx > 0}_{ly > 0}", "type": "box",
                "pos": f"{ux + lx} {ly} -.425", "size": ".025 .025 .325", "rgba": ".20 .12 .07 1",
            })
        for side, ax in (("left", -.26), ("right", .26)):
            prefix = f"u{unit}_{side}"
            arm_body = _prefixed(base, prefix)
            arm_body.set("pos", f"{ux + ax} .20 .0")
            for geom in arm_body.iter("geom"):
                if "jaw_pad_" in geom.attrib.get("name", ""):
                    geom.set("friction", PAD_FRICTION)
                    geom.set("solref", PAD_SOLREF)
                    geom.set("solimp", PAD_SOLIMP)
            worldbody.append(arm_body)
            for exclude in source_excludes:
                ET.SubElement(contact, "exclude", {
                    "body1": f"{prefix}_{exclude.attrib['body1']}", "body2": f"{prefix}_{exclude.attrib['body2']}",
                })
            for actuator in source_actuators:
                copied = copy.deepcopy(actuator)
                copied.set("name", f"{prefix}_{actuator.attrib['name']}")
                copied.set("joint", f"{prefix}_{actuator.attrib['joint']}")
                actuators.append(copied)
        # tableware for the unit: two cups on the left half, a plate on the right
        worldbody.append(_hollow_cup(f"u{unit}_cup_a", f"{ux - 0.10} 0.02 .045", rgba=".22 .58 .78 1", mass=".10",
                                     friction="3.00 .020 .001", contact={}))
        worldbody.append(_hollow_cup(f"u{unit}_cup_b", f"{ux - 0.20} -0.14 .045", rgba=".78 .40 .30 1", mass=".10",
                                     friction="3.00 .020 .001", contact={}))
        worldbody.append(_lipped_plate(f"u{unit}_plate", f"{ux + 0.12} -0.18 .0105", rgba=".92 .92 .90 1", mass=".20",
                                       friction="1.20 .006 .0002"))

    # -- the completely unnecessary drone (Skydio X2, kinematic) -------------
    if drone:
        ET.SubElement(asset, "texture", {"name": "x2_tex", "type": "2d", "file": "X2_lowpoly_texture.png"})
        ET.SubElement(asset, "material", {"name": "x2_mat", "texture": "x2_tex"})
        ET.SubElement(asset, "mesh", {"name": "x2_mesh", "file": "X2_lowpoly.obj", "scale": "0.01 0.01 0.01"})
        d = ET.SubElement(worldbody, "body", {"name": "drone_01", "mocap": "true", "pos": "%.3f %.3f %.3f" % DRONE_PAD})
        ET.SubElement(d, "geom", {"name": "drone_01_body", "type": "mesh", "mesh": "x2_mesh", "material": "x2_mat",
                                  "quat": "0 0 1 1", "contype": "0", "conaffinity": "0", "mass": "0"})
        for i, (rx, ry, rz) in enumerate(((-.14, -.18, .05), (-.14, .18, .05), (.14, .18, .08), (.14, -.18, .08))):
            ET.SubElement(d, "geom", {"name": f"drone_01_rotor{i}", "type": "ellipsoid", "size": ".13 .13 .004",
                                      "pos": f"{rx} {ry} {rz}", "rgba": ".15 .15 .15 .35", "contype": "0", "conaffinity": "0", "mass": "0"})
        ET.SubElement(worldbody, "geom", {"name": "drone_pad", "type": "cylinder", "size": ".30 .01",
                                          "pos": "%.3f %.3f %.3f" % (DRONE_PAD[0], DRONE_PAD[1], -0.75 + 0.01),
                                          "rgba": ".15 .35 .25 1"})

    # -- the mobile unit: a rover carrying a tray with a cup on it ----------
    if rover:
        r = ET.SubElement(worldbody, "body", {"name": "mobile_01", "mocap": "true", "pos": "%.3f %.3f %.3f" % ROVER_START})
        ET.SubElement(r, "geom", {"name": "mobile_01_chassis", "type": "box", "size": ".22 .15 .06", "pos": "0 0 .12",
                                  "rgba": ".85 .55 .15 1", "mass": "0"})
        for i, (wx, wy) in enumerate(((.15, .17), (-.15, .17), (.15, -.17), (-.15, -.17))):
            ET.SubElement(r, "geom", {"name": f"mobile_01_wheel{i}", "type": "cylinder", "size": ".07 .03",
                                      "pos": f"{wx} {wy} .07", "euler": "1.5708 0 0", "rgba": ".1 .1 .1 1",
                                      "contype": "0", "conaffinity": "0", "mass": "0"})
        ET.SubElement(r, "geom", {"name": "mobile_01_mast", "type": "cylinder", "size": ".02 .27", "pos": "0 0 .45",
                                  "rgba": ".6 .6 .62 1", "mass": "0"})
        ET.SubElement(r, "geom", {"name": "mobile_01_tray", "type": "box", "size": ".16 .12 .008", "pos": "0 0 .73",
                                  "rgba": ".35 .35 .38 1", "friction": "1.5 .01 .001", "mass": "0"})
        worldbody.append(_hollow_cup("mobile_01_cargo", f"{ROVER_START[0]} {ROVER_START[1]} {-0.75 + 0.738 + 0.045 + 0.002}",
                                     rgba=".95 .85 .25 1", mass=".10", friction="3.00 .020 .001", contact={}))
    return ET.tostring(root, encoding="unicode")


def fleet_assets() -> dict[str, bytes]:
    assets = {f"assets/{p.name}": p.read_bytes() for p in ARM_ASSETS.glob("*.stl")}
    if SKYDIO.exists():
        assets["assets/X2_lowpoly.obj"] = (SKYDIO / "X2_lowpoly.obj").read_bytes()
        assets["assets/X2_lowpoly_texture.png"] = (SKYDIO / "X2_lowpoly_texture_SpinningProps_1024.png").read_bytes()
    return assets


def install(units: int = 4, drone: bool = True, rover: bool = True) -> None:
    """Make every IntelTableWorld built in this process load the fleet scene.
    Process-local; the submission module on disk is untouched."""
    import mujoco

    def load(config=None):
        xml = extend_fleet_xml(intel_sim.dual_so101_xml(config), units=units, drone=drone and SKYDIO.exists(), rover=rover)
        return mujoco.MjModel.from_xml_string(xml, assets=fleet_assets())
    intel_sim.load_dual_so101_model = load
