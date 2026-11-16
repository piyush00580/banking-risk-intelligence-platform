"""Flask application factory for local and test use."""
from __future__ import annotations

from pathlib import Path

from flask import Flask

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(test_config: dict | None = None) -> Flask:
    """Create the API without loading models; inference loads artifacts on demand."""
    app = Flask(__name__)
    app.config.from_mapping(
        TESTING=False,
        SERVICE_NAME="Banking Risk Intelligence API",
        API_VERSION="1.0.0",
        CREDIT_ARTIFACT_PATH=str(PROJECT_ROOT / "models" / "credit_risk_xgb.joblib"),
        CREDIT_METADATA_PATH=str(PROJECT_ROOT / "models" / "credit_risk_metadata.json"),
        FRAUD_ARTIFACT_PATH=str(PROJECT_ROOT / "models" / "fraud_detection_xgb.joblib"),
    )
    app.json.sort_keys = False
    if test_config:
        app.config.update(test_config)

    from .routes import api, register_route_error_handlers
    app.register_blueprint(api)
    register_route_error_handlers(app)

    @app.errorhandler(Exception)
    def unexpected_error(_error):
        app.logger.exception("Unhandled API exception")
        from .serializers import error_response
        return error_response(500, "INTERNAL_ERROR", "An unexpected server error occurred.")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
