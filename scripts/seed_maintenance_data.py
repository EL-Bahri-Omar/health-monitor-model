#!/usr/bin/env python
"""
SmartFactory — Seed realistic maintenance data into MongoDB.

Creates realistic interventions with diagnoses, observations, and actions
based on real industrial maintenance patterns for:
  - Temperature problems (overheating, cooling system failures)
  - Vibration problems (bearing wear, misalignment, imbalance)
  - Current problems (overload, electrical faults, VFD issues)
  - RPM problems (belt wear, mechanical blockage, motor faults)

This data helps the AI model learn from past interventions to:
  1. Predict problems BEFORE they happen (20+ min early)
  2. Recommend specific fixes based on intervention history

Usage:
  python scripts/seed_maintenance_data.py --api-url http://localhost:8080
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone

import requests

# ─── Configuration ───────────────────────────────────────────────────────

API_KEY = "smartfactory-ai-internal-2024"
DEFAULT_API_URL = "http://localhost:8080"

# Machine codes that exist in the SmartFactory database
MACHINE_CODES = ["GEN-001", "CNC-002", "PMP-003", "CMP-004", "MTR-005"]

# ─── Realistic Intervention Templates ────────────────────────────────────

TEMPERATURE_INTERVENTIONS = [
    {
        "title": "Overheating — Cooling system failure",
        "description": "Machine temperature exceeded 75°C. Cooling system not functioning properly. Risk of thermal damage to motor windings.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Coolant level critically low — reservoir nearly empty",
            "Cooling fan motor seized due to bearing failure",
            "Heat exchanger fins blocked with dust and debris",
        ],
        "observations": [
            "Temperature reading: 78°C (normal: 40-55°C)",
            "Coolant level below minimum mark",
            "Fan motor not spinning — no airflow detected",
            "Heat discoloration visible on motor housing",
        ],
        "actions": [
            "Stopped machine immediately to prevent thermal damage",
            "Replaced cooling fan motor and bearings",
            "Refilled coolant reservoir with approved coolant",
            "Cleaned heat exchanger fins and air intake vents",
            "Verified temperature returned to 42°C after 30 min cool-down",
        ],
    },
    {
        "title": "High temperature trend — Air filter blockage",
        "description": "Gradual temperature rise from 45°C to 62°C over 4 hours. Air circulation restricted.",
        "type": "PREVENTIVE",
        "preventiveSubtype": "CONDITIONAL",
        "priority": "MEDIUM",
        "diagnoses": [
            "Air intake filters clogged with industrial dust",
            "Ventilation duct partially obstructed by packaging material",
        ],
        "observations": [
            "Temperature rising steadily — 45°C → 55°C → 62°C",
            "Reduced airflow at ventilation outlets",
            "Air filters dark brown/black (should be white/gray)",
        ],
        "actions": [
            "Replaced all air intake filters",
            "Cleared ventilation duct obstruction",
            "Cleaned fan blades and housing",
            "Scheduled quarterly filter replacement",
            "Temperature stabilized at 44°C after maintenance",
        ],
    },
    {
        "title": "Motor winding overheating — Insulation degradation",
        "description": "Motor running hot at 70°C with intermittent current spikes. Thermal imaging shows hot spots on stator windings.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Insulation resistance below minimum (< 1 MΩ at 500V DC)",
            "Phase-to-phase insulation breakdown beginning",
            "Motor overloaded beyond rated capacity",
        ],
        "observations": [
            "Temperature at motor surface: 70°C",
            "Megger test: insulation resistance = 0.8 MΩ (min required: 2 MΩ)",
            "Current draw 15% above nameplate rating",
            "Thermal camera shows hot spot on B-phase winding",
        ],
        "actions": [
            "Reduced production load to 80% of rated capacity",
            "Applied varnish re-impregnation to motor windings",
            "Replaced damaged phase connection terminals",
            "Re-tested insulation resistance: 5.2 MΩ (pass)",
            "Programmed thermal overload relay to trip at 65°C",
        ],
    },
]

VIBRATION_INTERVENTIONS = [
    {
        "title": "Excessive vibration — Bearing replacement",
        "description": "Vibration levels at 7.5 mm/s RMS. Bearing defect frequency detected in spectrum analysis.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Inner race defect on drive-end bearing (SKF 6308)",
            "Bearing cage showing metal fatigue and flaking",
            "Insufficient lubrication — grease dried out",
        ],
        "observations": [
            "Vibration RMS: 7.5 mm/s (alarm threshold: 5.0 mm/s)",
            "Spectrum analysis: BPFI frequency spike at 128 Hz",
            "Audible grinding noise from drive-end housing",
            "Bearing temperature: 65°C (normal: 40°C)",
        ],
        "actions": [
            "Replaced both drive-end and non-drive-end bearings",
            "Cleaned bearing housing and shaft journal",
            "Applied correct grade/quantity of bearing grease",
            "Verified vibration reduced to 1.8 mm/s RMS post-repair",
            "Added bearing to quarterly vibration monitoring route",
        ],
    },
    {
        "title": "Vibration spike — Shaft misalignment",
        "description": "2x running speed vibration dominant. Angular misalignment between motor and driven equipment.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Angular misalignment: 0.12mm/100mm (limit: 0.05mm/100mm)",
            "Coupling spider insert worn and deformed",
            "Soft foot on motor — front-left bolt not making contact",
        ],
        "observations": [
            "Vibration spectrum: dominant 2x peak at 50 Hz",
            "Axial vibration elevated: 4.2 mm/s",
            "Coupling spider shows uneven compression marks",
            "Gap gauge shows 0.15mm gap under motor front-left foot",
        ],
        "actions": [
            "Corrected soft foot with precision shims",
            "Performed laser shaft alignment (final: 0.02mm/100mm)",
            "Replaced worn jaw coupling spider insert",
            "Verified vibration dropped to 1.5 mm/s overall",
            "Documented alignment values for future reference",
        ],
    },
    {
        "title": "Increasing vibration — Rotating imbalance",
        "description": "Vibration trending upward: 3.0 to 5.5 mm/s over 2 weeks. 1x running speed dominant.",
        "type": "PREVENTIVE",
        "preventiveSubtype": "CONDITIONAL",
        "priority": "MEDIUM",
        "diagnoses": [
            "Mass imbalance on impeller/rotor due to material buildup",
            "Fan blade damaged — chip missing from leading edge",
        ],
        "observations": [
            "1x running speed amplitude growing: 3.0 → 4.2 → 5.5 mm/s",
            "Phase angle stable — confirms mass imbalance",
            "Visual: material buildup on impeller blades",
        ],
        "actions": [
            "Cleaned impeller/rotor — removed 45g of deposited material",
            "Performed two-plane trim balance",
            "Final residual imbalance: 0.8 mm/s at 1x (within ISO 1940 G2.5)",
            "Scheduled monthly impeller inspection",
        ],
    },
]

CURRENT_INTERVENTIONS = [
    {
        "title": "Current overload — Mechanical binding",
        "description": "Current draw 30% above normal. Motor struggling against increased mechanical resistance.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Gearbox output shaft binding due to damaged gear teeth",
            "Excessive load from downstream process backup",
            "Lubrication failure in gearbox — oil level critical",
        ],
        "observations": [
            "Motor current: 12.5A (rated: 9.0A, normal: 8.2A)",
            "Gearbox oil level below sight glass minimum",
            "Metal particles visible in gearbox oil sample",
            "Machine RPM dropping under load",
        ],
        "actions": [
            "Drained gearbox oil — heavy metallic contamination found",
            "Inspected gears: replaced damaged intermediate gear set",
            "Flushed gearbox housing and refilled with fresh EP220 gear oil",
            "Installed oil level sensor with low-level alarm",
            "Current returned to 8.5A normal operation",
        ],
    },
    {
        "title": "Current fluctuation — VFD malfunction",
        "description": "Erratic current readings with ±3A swings. Variable frequency drive (VFD) output unstable.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "VFD output IGBT module partially failed (Phase U)",
            "DC bus capacitor degraded — ripple current excessive",
            "Control board firmware fault — incorrect PID tuning",
        ],
        "observations": [
            "Current swinging between 6A and 13A erratically",
            "VFD fault log: multiple overcurrent trips",
            "Motor running rough with audible hunting",
            "DC bus voltage ripple: 18% (max: 5%)",
        ],
        "actions": [
            "Replaced failed IGBT power module on Phase U",
            "Replaced DC bus electrolytic capacitors (pair)",
            "Updated VFD firmware to latest version",
            "Re-tuned motor auto-tune parameters",
            "Verified smooth current profile at all speed setpoints",
        ],
    },
    {
        "title": "Elevated current — Loose electrical connections",
        "description": "Current 15% above normal with temperature rising. Thermal imaging shows hot terminal connections.",
        "type": "PREVENTIVE",
        "preventiveSubtype": "CONDITIONAL",
        "priority": "MEDIUM",
        "diagnoses": [
            "Terminal connections at motor junction box loosened by vibration",
            "Oxidation on power cable crimps increasing contact resistance",
            "Phase B terminal lug torque below specification",
        ],
        "observations": [
            "Current draw: 9.8A (normal: 8.2A)",
            "Thermal imaging: Phase B terminal 85°C (others: 45°C)",
            "Terminal bolt torque: 3 Nm (specification: 8 Nm)",
            "Light arcing marks visible on terminal surface",
        ],
        "actions": [
            "De-energized and locked out motor circuit",
            "Cleaned all terminal lugs with contact cleaner",
            "Re-crimped cable ends with new copper lugs",
            "Torqued all terminals to specification (8 Nm)",
            "Applied anti-oxidant compound to connections",
            "Current reduced to 8.1A — thermal scan normal",
        ],
    },
]

RPM_INTERVENTIONS = [
    {
        "title": "RPM drop — Drive belt worn",
        "description": "Machine RPM dropped from 1500 to 1350. Belt slipping under load.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "V-belt stretched beyond tensioning range",
            "Belt sidewalls glazed and cracked — loss of friction",
            "Drive pulley groove worn — belt riding too deep",
        ],
        "observations": [
            "RPM reading: 1350 (rated: 1500)",
            "Belt visibly loose — excessive deflection under thumb pressure",
            "Belt surface shiny/glazed with hairline cracks",
            "Squealing noise on startup and load changes",
        ],
        "actions": [
            "Replaced all drive belts (matched set of 3)",
            "Inspected pulleys — replaced worn driven pulley",
            "Set belt tension to manufacturer specification",
            "Verified RPM returned to 1495 under full load",
            "Scheduled belt inspection every 3 months",
        ],
    },
    {
        "title": "RPM fluctuation — Motor stall risk",
        "description": "RPM unstable between 1200-1450 with current spikes. Risk of motor stall.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Motor rotor bar cracks detected (broken rotor bar test)",
            "Stator winding inter-turn fault developing",
            "Power supply voltage drop during peak production hours",
        ],
        "observations": [
            "RPM fluctuating: 1200 to 1450 over 10 seconds",
            "Current spiking to 16A during RPM dips",
            "Motor making uneven humming noise",
            "Supply voltage: 375V (nominal: 400V) — 6.25% under-voltage",
        ],
        "actions": [
            "Installed power factor correction capacitor bank",
            "Replaced motor with new unit (original had 2 broken rotor bars)",
            "Checked and tightened all MCC bus bar connections",
            "Verified stable RPM: 1498 ± 5 under full load",
            "Added under-voltage relay protection to motor circuit",
        ],
    },
    {
        "title": "Gradual RPM decline — Clutch wear",
        "description": "RPM gradually decreasing: 1500 → 1480 → 1450 over 2 weeks. Mechanical slip increasing.",
        "type": "PREVENTIVE",
        "preventiveSubtype": "CONDITIONAL",
        "priority": "MEDIUM",
        "diagnoses": [
            "Clutch friction plates worn beyond service limit",
            "Clutch pressure spring weakened — insufficient clamping force",
        ],
        "observations": [
            "RPM trend: 1500 → 1480 → 1460 → 1450 over 14 days",
            "Clutch slipping under high load — RPM recovers at no-load",
            "Clutch friction material thickness: 2mm (min: 4mm)",
        ],
        "actions": [
            "Replaced clutch friction plates (new: 8mm thickness)",
            "Replaced clutch pressure springs",
            "Adjusted clutch engagement pressure to specification",
            "RPM stable at 1498 under full load",
            "Scheduled clutch inspection every 6 months",
        ],
    },
]

MULTI_SENSOR_INTERVENTIONS = [
    {
        "title": "Combined — Temperature + Vibration (Bearing failure)",
        "description": "Temperature 72°C with vibration 6.8 mm/s. Classic bearing failure signature.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Bearing outer race spalling causing friction heat and vibration",
            "Lubrication system failed — grease line blocked",
            "Bearing seal damaged — contamination ingress",
        ],
        "observations": [
            "Temperature: 72°C (normal: 42°C), Vibration: 6.8 mm/s (normal: 2.0 mm/s)",
            "Bearing defect frequencies confirmed in vibration spectrum",
            "Grease around bearing seal is dark and contains metal particles",
        ],
        "actions": [
            "Emergency stop — machine shut down for bearing replacement",
            "Replaced both bearings with sealed type (2RS) to prevent contamination",
            "Cleared and replaced automatic grease lubrication line",
            "Cleaned shaft journal to remove scoring marks",
            "Post-repair: temp 40°C, vibration 1.6 mm/s",
        ],
    },
    {
        "title": "Combined — Current + RPM (Motor overload)",
        "description": "Current at 13A with RPM dropping to 1380. Motor overloaded by downstream process.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Conveyor belt jam causing excessive torque demand",
            "Product accumulation at discharge point",
            "Motor undersized for actual process requirements",
        ],
        "observations": [
            "Current: 13A (rated: 9A), RPM: 1380 (rated: 1500)",
            "Conveyor belt visibly strained — material piling up at discharge",
            "Motor frame hot to touch",
        ],
        "actions": [
            "Cleared material jam at conveyor discharge point",
            "Installed jam detection sensor with auto-stop",
            "Upgraded motor from 5.5kW to 7.5kW for adequate margin",
            "Post-repair: current 7.8A, RPM 1498",
        ],
    },
    {
        "title": "Combined — Temperature + Current (Electrical fault)",
        "description": "Temperature 68°C with current 11.5A. Possible stator winding insulation issue.",
        "type": "CORRECTIVE",
        "priority": "HIGH",
        "diagnoses": [
            "Inter-turn short in stator winding Phase C",
            "Winding insulation degraded by prolonged overheating",
        ],
        "observations": [
            "Temperature: 68°C, Current: 11.5A (asymmetric across phases)",
            "Phase C current 18% higher than Phase A and B",
            "Insulation resistance test: Phase C to ground = 1.2 MΩ (others: 8+ MΩ)",
        ],
        "actions": [
            "Rewound Phase C stator winding",
            "Applied Class H insulation varnish to all windings",
            "Installed winding RTD temperature sensors for continuous monitoring",
            "Post-repair: balanced phase currents, temp 41°C",
        ],
    },
]

ALL_INTERVENTION_TEMPLATES = (
    TEMPERATURE_INTERVENTIONS
    + VIBRATION_INTERVENTIONS
    + CURRENT_INTERVENTIONS
    + RPM_INTERVENTIONS
    + MULTI_SENSOR_INTERVENTIONS
)


def _get_auth_token(api_url: str) -> str:
    """Login with admin credentials and get JWT token."""
    login_url = f"{api_url}/api/auth/login"
    creds = {"email": "admin@smartfactory.com", "password": "admin123"}
    resp = requests.post(login_url, json=creds, timeout=10)
    resp.raise_for_status()
    return resp.json().get("token") or resp.json().get("accessToken")


def _get_machines(api_url: str, headers: dict) -> list[dict]:
    """Fetch all machines from the backend."""
    resp = requests.get(f"{api_url}/api/machines?size=500", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("content", data) if isinstance(data, dict) else data


def _create_intervention(api_url: str, headers: dict, body: dict) -> dict:
    """Create a maintenance intervention via API."""
    resp = requests.post(f"{api_url}/api/maintenance", json=body, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _add_diagnosis(api_url: str, headers: dict, intervention_id: str, text: str):
    """Add a diagnosis to an intervention."""
    resp = requests.post(
        f"{api_url}/api/maintenance/{intervention_id}/diagnoses",
        json={"description": text},
        headers=headers,
        timeout=10,
    )
    if not resp.ok:
        print(f"  [WARN] diagnosis POST failed ({resp.status_code}): {text[:60]}")


def _add_action(api_url: str, headers: dict, intervention_id: str, text: str):
    """Add an action to an intervention."""
    resp = requests.post(
        f"{api_url}/api/maintenance/{intervention_id}/actions",
        json={"description": text},
        headers=headers,
        timeout=10,
    )
    if not resp.ok:
        print(f"  [WARN] action POST failed ({resp.status_code}): {text[:60]}")


def _add_observation(api_url: str, headers: dict, intervention_id: str, text: str):
    """Add an observation to an intervention."""
    resp = requests.post(
        f"{api_url}/api/maintenance/{intervention_id}/observations",
        json={"description": text},
        headers=headers,
        timeout=10,
    )
    if not resp.ok:
        print(f"  [WARN] observation POST failed ({resp.status_code}): {text[:60]}")


def seed_interventions(api_url: str):
    """Seed the database with realistic maintenance interventions."""
    print(f"\n{'='*60}")
    print(f"  SmartFactory — Maintenance Data Seeder")
    print(f"  API: {api_url}")
    print(f"{'='*60}\n")

    # Authenticate
    print("[1/4] Authenticating...")
    token = _get_auth_token(api_url)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    print(f"  ✓ Authenticated")

    # Get machines
    print("[2/4] Fetching machines...")
    machines = _get_machines(api_url, headers)
    if not machines:
        print("  ✗ No machines found! Create machines first.")
        return
    machine_map = {}
    for m in machines:
        code = m.get("code", "")
        machine_map[code] = m
        print(f"  Machine: {code} — {m.get('name', '?')} (id: {m.get('id', '?')[:8]}...)")

    # Seed interventions
    print(f"\n[3/4] Creating {len(ALL_INTERVENTION_TEMPLATES)} intervention templates × multiple machines...")
    created = 0
    base_date = datetime.now(timezone.utc) - timedelta(days=60)

    for i, template in enumerate(ALL_INTERVENTION_TEMPLATES):
        # Pick 1-3 machines for each template
        num_machines = random.randint(1, min(3, len(machines)))
        selected = random.sample(machines, num_machines)

        for machine in selected:
            # Randomize date within the last 60 days
            days_offset = random.randint(0, 55)
            hours_offset = random.randint(6, 20)
            timestamp = base_date + timedelta(days=days_offset, hours=hours_offset)

            body = {
                "machineId": machine.get("id") or machine.get("code"),
                "title": template["title"],
                "description": template["description"],
                "type": template.get("type", "CORRECTIVE"),
                "priority": template.get("priority", "MEDIUM"),
            }
            if template.get("preventiveSubtype"):
                body["preventiveSubtype"] = template["preventiveSubtype"]

            try:
                result = _create_intervention(api_url, headers, body)
                intervention_id = result.get("id")
                code = machine.get("code", "?")
                print(f"  [{created+1}] Created: {template['title'][:50]}... → {code}")

                if intervention_id:
                    # Add diagnoses
                    for diag in template.get("diagnoses", []):
                        _add_diagnosis(api_url, headers, intervention_id, diag)

                    # Add observations
                    for obs in template.get("observations", []):
                        _add_observation(api_url, headers, intervention_id, obs)

                    # Add actions
                    for act in template.get("actions", []):
                        _add_action(api_url, headers, intervention_id, act)

                created += 1
            except Exception as e:
                print(f"  [ERROR] Failed to create: {template['title'][:50]}... — {e}")

    print(f"\n[4/4] Summary")
    print(f"  ✓ Created {created} interventions with diagnoses, observations, and actions")
    print(f"  Templates: {len(ALL_INTERVENTION_TEMPLATES)}")
    print(f"  Machines: {len(machines)}")
    print(f"\n  The AI model can now learn from these interventions to:")
    print(f"    • Predict problems 20+ min before they happen")
    print(f"    • Recommend specific fixes based on intervention history")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed SmartFactory with maintenance data")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Backend API URL")
    args = parser.parse_args()
    seed_interventions(args.api_url)

