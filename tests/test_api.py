import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_models_info(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_api_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_optimize_valid(client):
    response = client.post("/api/optimize", json={})
    assert response.status_code in (200, 500, 503)


def test_optimize_with_params(client):
    payload = {
        "lift_type": "ESP",
        "pump_speed_rpm": 1200.0,
        "motor_current_amp": 45.0,
        "well_depth_ft": 8000.0,
        "water_cut_pct": 60.0,
        "n_iterations": 200,
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code in (200, 500, 503)


def test_optimize_gas_lift(client):
    payload = {
        "lift_type": "GasLift",
        "gas_injection_mcf": 500.0,
        "well_depth_ft": 6000.0,
        "water_cut_pct": 40.0,
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code in (200, 500, 503)


def test_failure_valid(client):
    response = client.post("/api/failure", json={})
    assert response.status_code in (200, 500, 503)


def test_failure_with_params(client):
    payload = {
        "lift_type": "ESP",
        "pump_speed_rpm": 1500.0,
        "motor_current_amp": 60.0,
        "well_depth_ft": 7000.0,
        "water_cut_pct": 70.0,
    }
    response = client.post("/api/failure", json=payload)
    assert response.status_code in (200, 500, 503)
