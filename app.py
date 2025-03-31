from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from yield_data_fetcher import YieldDataFetcher
from yield_analyzer import YieldAnalyzer
import json
from datetime import datetime, timedelta
import os
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes

# Initialize data fetcher and analyzer
data_fetcher = YieldDataFetcher()
analyzer = YieldAnalyzer()

# Cached data to avoid frequent API calls
cached_data = {
    "full_data": None,
    "latest_data": None,
    "cycle_data": {},
    "last_update": None,
}


def refresh_data(force=False):
    """Refresh cached data if needed"""
    current_time = datetime.now()

    # If last update was less than 6 hours ago and not forced, use cached data
    if cached_data["last_update"] and not force:
        time_diff = (current_time - cached_data["last_update"]).total_seconds() / 3600
        if time_diff < 6:
            return

    # Fetch and cache data
    try:
        print("Refreshing data cache...")
        cached_data["full_data"] = data_fetcher.fetch_historical_data()
        cached_data["latest_data"] = data_fetcher.fetch_latest_data(
            days=180
        )  # Get more data

        # Fetch data for each cutting cycle
        for cycle_name in data_fetcher.cutting_cycles.keys():
            cached_data["cycle_data"][cycle_name] = data_fetcher.fetch_cycle_data(
                cycle_name
            )

        cached_data["last_update"] = current_time
        print("Data cache successfully refreshed")
    except Exception as e:
        print(f"Error refreshing data cache: {str(e)}")


# Helper function to convert DataFrame to JSON-compatible format
def dataframe_to_json(df):
    if df is None or df.empty:
        return []

    # Convert DataFrame to dict with date as string
    result = []
    for date, row in df.iterrows():
        row_dict = {"date": date.strftime("%Y-%m-%d")}
        for col, value in row.items():
            # Handle NaN values
            if pd.isna(value):
                row_dict[col] = None
            else:
                # Convert to percentage for yield values
                if col in analyzer.tenors or col in analyzer.spreads:
                    row_dict[col] = round(
                        float(value) * 100, 2
                    )  # Convert to percentage with 2 decimal places
                else:
                    row_dict[col] = value
        result.append(row_dict)

    return result


# Main routes
@app.route("/")
def index():
    """Main dashboard page"""
    return render_template("index.html")


@app.route("/cycles")
def cycles_page():
    """Cutting cycles analysis page"""
    return render_template("cycles.html")


@app.route("/analysis")
def analysis_page():
    """Custom analysis page"""
    return render_template("analysis.html")


# API routes
@app.route("/api/health", methods=["GET"])
def health_check():
    """API health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/data/yields", methods=["GET"])
def get_yield_data():
    """Get historical yield data"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if cached_data["full_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["full_data"].copy()

    # Filter by date if provided
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        except:
            return jsonify({"error": "Invalid start_date format"}), 400

    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        except:
            return jsonify({"error": "Invalid end_date format"}), 400

    # Convert to JSON format
    json_data = dataframe_to_json(data)

    return jsonify({"data": json_data, "count": len(json_data)})


@app.route("/api/data/latest", methods=["GET"])
def get_latest_data():
    """Get latest yield data"""
    refresh_data()

    days = request.args.get("days", default=90, type=int)

    if cached_data["latest_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["latest_data"].copy()

    # Convert to JSON format
    json_data = dataframe_to_json(data)

    return jsonify(
        {
            "data": json_data,
            "count": len(json_data),
            "tenors": analyzer.tenors,
            "spreads": analyzer.spreads,
        }
    )


@app.route("/api/cycles", methods=["GET"])
def get_cutting_cycles():
    """Get list of cutting cycles"""
    cycles = []

    for name, details in data_fetcher.cutting_cycles.items():
        cycles.append(
            {
                "id": name,
                "name": details["description"],
                "start_date": details["start"],
                "end_date": details["end"],
            }
        )

    return jsonify({"cycles": cycles})


@app.route("/api/analysis/cycle/<cycle_name>", methods=["GET"])
def analyze_cycle(cycle_name):
    """Analyze a specific cutting cycle"""
    refresh_data()

    if cycle_name not in data_fetcher.cutting_cycles:
        return jsonify({"error": f"Unknown cycle: {cycle_name}"}), 400

    if (
        cycle_name not in cached_data["cycle_data"]
        or cached_data["cycle_data"][cycle_name] is None
    ):
        return (
            jsonify({"error": "Cycle data not available. Please try again later."}),
            500,
        )

    cycle_data = cached_data["cycle_data"][cycle_name].copy()
    cycle_info = data_fetcher.cutting_cycles[cycle_name]

    # Perform analysis
    try:
        analysis = analyzer.analyze_cycle(
            cycle_data, cycle_start=cycle_info["start"], cycle_end=cycle_info["end"]
        )

        # Generate report
        report = analyzer.generate_report(cycle_data, cycle_info["description"])

        # Add cycle metadata
        analysis["cycle_info"] = {
            "name": cycle_info["description"],
            "start_date": cycle_info["start"],
            "end_date": cycle_info["end"],
            "report": report,
        }

        # Add data for charts (convert to percentage)
        analysis["data"] = dataframe_to_json(cycle_data)

        # Add PCA decomposition
        pca_results = analyzer.decompose_yields(cycle_data)
        if pca_results and "components" in pca_results:
            analysis["pca"] = {
                "explained_variance": [
                    round(v * 100, 2) for v in pca_results["explained_variance"]
                ],  # As percentages
                "components": dataframe_to_json(pca_results["components"]),
                "loadings": pca_results["loadings"].to_dict(),
            }

        # Detect regime changes
        regime_changes = analyzer.detect_regime_changes(cycle_data)
        formatted_regimes = []
        for regime in regime_changes:
            formatted_regimes.append(
                {
                    "date": regime["date"].strftime("%Y-%m-%d"),
                    "direction": regime["direction"],
                    "magnitude": round(
                        float(regime["magnitude"]) * 100, 2
                    ),  # As percentage
                    "tenor": regime["tenor"],
                }
            )

        analysis["regimes"] = formatted_regimes

        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500


@app.route("/api/analysis/custom", methods=["POST"])
def custom_analysis():
    """Perform custom analysis on specified date range"""
    refresh_data()

    try:
        # Get parameters from request
        params = request.json
        start_date = params.get("start_date")
        end_date = params.get("end_date")

        if not start_date or not end_date:
            return jsonify({"error": "Start date and end date are required"}), 400

        # Get data
        data = cached_data["full_data"].copy()

        # Filter by date
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        except Exception as e:
            return jsonify({"error": f"Date filtering error: {str(e)}"}), 400

        if filtered_data.empty:
            return (
                jsonify({"error": "No data available for the specified date range"}),
                404,
            )

        # Perform analysis
        analysis = analyzer.analyze_cycle(filtered_data)

        # Generate report
        report = analyzer.generate_report(filtered_data, "Custom Analysis")

        # Add metadata
        analysis["info"] = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "report": report,
        }

        # Add data for charts
        analysis["data"] = dataframe_to_json(filtered_data)

        # Add PCA decomposition
        pca_results = analyzer.decompose_yields(filtered_data)
        if pca_results and "components" in pca_results:
            analysis["pca"] = {
                "explained_variance": [
                    round(v * 100, 2) for v in pca_results["explained_variance"]
                ],  # As percentages
                "components": dataframe_to_json(pca_results["components"]),
                "loadings": pca_results["loadings"].to_dict(),
            }

        # Detect regime changes
        regime_changes = analyzer.detect_regime_changes(filtered_data)
        formatted_regimes = []
        for regime in regime_changes:
            formatted_regimes.append(
                {
                    "date": regime["date"].strftime("%Y-%m-%d"),
                    "direction": regime["direction"],
                    "magnitude": round(
                        float(regime["magnitude"]) * 100, 2
                    ),  # As percentage
                    "tenor": regime["tenor"],
                }
            )

        analysis["regimes"] = formatted_regimes

        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500


@app.route("/api/charts/yields", methods=["GET"])
def get_yields_chart():
    """Generate yields chart for specified date range or cycle"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    cycle = request.args.get("cycle")

    # If cycle is specified, use cycle data
    if cycle and cycle in cached_data["cycle_data"]:
        data = cached_data["cycle_data"][cycle].copy()
    elif cached_data["full_data"] is not None:
        data = cached_data["full_data"].copy()

        # Filter by date if provided
        if start_date:
            try:
                start_date = pd.to_datetime(start_date)
                data = data[data.index >= start_date]
            except:
                return jsonify({"error": "Invalid start_date format"}), 400

        if end_date:
            try:
                end_date = pd.to_datetime(end_date)
                data = data[data.index <= end_date]
            except:
                return jsonify({"error": "Invalid end_date format"}), 400
    else:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    # Generate chart
    try:
        title = (
            f"Treasury Yields - {cycle.replace('_', ' ').title()}"
            if cycle
            else "Treasury Yields"
        )
        fig = analyzer.plot_yields(data, title)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": chart_json})
    except Exception as e:
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


@app.route("/api/charts/pca", methods=["GET"])
def get_pca_chart():
    """Generate PCA chart for specified date range"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if cached_data["full_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["full_data"].copy()

    # Filter by date if provided
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        except:
            return jsonify({"error": "Invalid start_date format"}), 400

    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        except:
            return jsonify({"error": "Invalid end_date format"}), 400

    # Generate chart
    try:
        fig = analyzer.plot_pca_components(data)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": chart_json})
    except Exception as e:
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


@app.route("/api/charts/regimes", methods=["GET"])
def get_regimes_chart():
    """Generate regime changes chart for specified date range"""
    refresh_data()

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    tenor = request.args.get("tenor", default="10Y")

    if cached_data["full_data"] is None:
        return jsonify({"error": "Data not available. Please try again later."}), 500

    data = cached_data["full_data"].copy()

    # Filter by date if provided
    if start_date:
        try:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        except:
            return jsonify({"error": "Invalid start_date format"}), 400

    if end_date:
        try:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        except:
            return jsonify({"error": "Invalid end_date format"}), 400

    # Detect regime changes
    regime_changes = analyzer.detect_regime_changes(data, tenor=tenor)

    # Generate chart
    try:
        fig = analyzer.plot_regime_changes(data, regime_changes)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"chart": chart_json})
    except Exception as e:
        return jsonify({"error": f"Chart generation error: {str(e)}"}), 500


@app.route("/api/refresh", methods=["POST"])
def force_refresh():
    """Force refresh of cached data"""
    try:
        refresh_data(force=True)
        return jsonify(
            {
                "status": "success",
                "message": "Data cache refreshed",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Error refreshing data: {str(e)}"}),
            500,
        )


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    # Initial data load
    refresh_data(force=True)

    # Run the app
    app.run(debug=True, host="0.0.0.0", port=5000)
