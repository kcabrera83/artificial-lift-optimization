import unittest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["models_loaded"]["optimizer"])
        self.assertTrue(data["models_loaded"]["failure_predictor"])

    def test_models_endpoint(self):
        resp = self.client.get("/api/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("optimizer", data)
        self.assertIn("failure_predictor", data)
        self.assertIn("r2", data["optimizer"])
        self.assertIn("accuracy", data["failure_predictor"])

    def test_optimize_esp(self):
        payload = {
            "lift_type": "ESP",
            "pump_speed_rpm": 2500,
            "downhole_pressure_psi": 2000,
            "motor_current_amp": 40,
            "well_depth_ft": 6000,
            "water_cut_pct": 50,
        }
        resp = self.client.post(
            "/api/optimize",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["lift_type"], "ESP")
        self.assertIn("optimal_params", data)
        self.assertIn("predicted_production_bbl_d", data)
        self.assertIn("current_production_bbl_d", data)
        self.assertGreater(data["predicted_production_bbl_d"], 0)

    def test_optimize_rod_pump(self):
        payload = {
            "lift_type": "rod_pump",
            "pump_speed_rpm": 6,
            "rod_load_klbf": 25,
            "downhole_pressure_psi": 1500,
            "well_depth_ft": 4000,
            "water_cut_pct": 40,
        }
        resp = self.client.post(
            "/api/optimize",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["lift_type"], "rod_pump")

    def test_optimize_gas_lift(self):
        payload = {
            "lift_type": "gas_lift",
            "gas_injection_mcf": 800,
            "downhole_pressure_psi": 2500,
            "well_depth_ft": 8000,
            "water_cut_pct": 60,
        }
        resp = self.client.post(
            "/api/optimize",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["lift_type"], "gas_lift")

    def test_failure_normal(self):
        payload = {
            "lift_type": "ESP",
            "pump_speed_rpm": 2800,
            "rod_load_klbf": 0,
            "gas_injection_mcf": 0,
            "downhole_pressure_psi": 1500,
            "motor_current_amp": 35,
            "well_depth_ft": 5000,
            "water_cut_pct": 40,
        }
        resp = self.client.post(
            "/api/failure",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("failure_mode", data)
        self.assertIn("confidence", data)
        self.assertIn(data["failure_mode"], [
            "normal", "pump_wear", "gas_lock", "rod_fatigue", "motor_overheat"
        ])

    def test_failure_overheat(self):
        payload = {
            "lift_type": "ESP",
            "pump_speed_rpm": 3200,
            "rod_load_klbf": 0,
            "gas_injection_mcf": 0,
            "downhole_pressure_psi": 3000,
            "motor_current_amp": 75,
            "well_depth_ft": 8000,
            "water_cut_pct": 80,
        }
        resp = self.client.post(
            "/api/failure",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("failure_mode", data)

    def test_failure_rod_fatigue(self):
        payload = {
            "lift_type": "rod_pump",
            "pump_speed_rpm": 8,
            "rod_load_klbf": 43,
            "gas_injection_mcf": 0,
            "downhole_pressure_psi": 2500,
            "motor_current_amp": 45,
            "well_depth_ft": 7000,
            "water_cut_pct": 70,
        }
        resp = self.client.post(
            "/api/failure",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("failure_mode", data)

    def test_optimize_no_body(self):
        resp = self.client.post("/api/optimize")
        self.assertEqual(resp.status_code, 400)

    def test_failure_no_body(self):
        resp = self.client.post("/api/failure")
        self.assertEqual(resp.status_code, 400)

    def test_index_page(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Artificial Lift Optimization", resp.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
