import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os


class YieldAnalyzer:
    """
    Analyzes Treasury yield data to identify patterns, calculate metrics,
    and detect regime changes during Fed cutting cycles.
    """

    def __init__(self):
        self.tenors = ["3M", "2Y", "5Y", "10Y", "30Y"]
        self.spreads = ["2s10s_spread", "5s10s_spread", "10s30s_spread"]
        # Define colors for consistent plotting
        self.colors = {
            "3M": "#1f77b4",  # blue
            "2Y": "#ff7f0e",  # orange
            "5Y": "#2ca02c",  # green
            "10Y": "#d62728",  # red
            "30Y": "#9467bd",  # purple
            "2s10s_spread": "#8c564b",  # brown
            "5s10s_spread": "#e377c2",  # pink
            "10s30s_spread": "#7f7f7f",  # gray
        }
        # Create plots directory if it doesn't exist
        self.plots_dir = r"C:\Users\Alicia_Work\Desktop\CISC_PRJ\plots"
        os.makedirs(self.plots_dir, exist_ok=True)

    def analyze_cycle(self, data, cycle_start=None, cycle_end=None):
        """
        Analyze yield behavior during a cutting cycle

        Args:
            data (DataFrame): Yield data
            cycle_start (str): Start date of cycle (YYYY-MM-DD)
            cycle_end (str): End date of cycle (YYYY-MM-DD)

        Returns:
            dict: Analysis results
        """
        # If cycle dates are provided, filter the data
        if cycle_start and cycle_end:
            cycle_data = data.loc[cycle_start:cycle_end].copy()
        else:
            # Use the entire dataset
            cycle_data = data.copy()

        # Get the tenors that are in the data
        available_tenors = [col for col in cycle_data.columns if col in self.tenors]

        # Check if we have enough data for full analysis
        if len(cycle_data) < 10:
            print(
                f"Warning: Limited data available ({len(cycle_data)} data points). Some analyses will be simplified."
            )

            # Perform limited analysis for small datasets
            analysis = {
                "yield_changes": self._calculate_yield_changes(
                    cycle_data, available_tenors
                ),
                "curve_characteristics": self._analyze_curve_shape(
                    cycle_data, available_tenors
                ),
                "summary_stats": self._calculate_summary_stats(
                    cycle_data, available_tenors
                ),
            }

            # Only add correlation if we have at least 3 data points
            if len(cycle_data) >= 3:
                analysis["correlation"] = self._calculate_correlations(
                    cycle_data, available_tenors
                )

            # Skip stationarity tests for small datasets
        else:
            # Perform full analysis
            analysis = {
                "yield_changes": self._calculate_yield_changes(
                    cycle_data, available_tenors
                ),
                "curve_characteristics": self._analyze_curve_shape(
                    cycle_data, available_tenors
                ),
                "summary_stats": self._calculate_summary_stats(
                    cycle_data, available_tenors
                ),
                "correlation": self._calculate_correlations(
                    cycle_data, available_tenors
                ),
                "stationarity": self._test_stationarity(cycle_data, available_tenors),
            }

        return analysis

    def _calculate_yield_changes(self, data, tenors):
        """Calculate yield changes during the period"""
        if len(data) < 2:
            return {}

        start_date = data.index[0]
        end_date = data.index[-1]

        changes = {}
        for tenor in tenors:
            if tenor in data.columns:
                start_yield = data.loc[start_date, tenor]
                end_yield = data.loc[end_date, tenor]
                abs_change = end_yield - start_yield
                pct_change = (abs_change / start_yield) * 100 if start_yield != 0 else 0

                changes[tenor] = {
                    "start_yield": start_yield,
                    "end_yield": end_yield,
                    "absolute_change": abs_change,
                    "percentage_change": pct_change,
                }

        return changes

    def _analyze_curve_shape(self, data, tenors):
        """Analyze yield curve shape and characteristics"""
        if len(data) < 2 or len(tenors) < 2:
            return {}

        start_date = data.index[0]
        end_date = data.index[-1]

        # Calculate key spreads at start and end
        spreads = {}

        # 2s10s spread (commonly watched)
        if "2Y" in data.columns and "10Y" in data.columns:
            start_2s10s = data.loc[start_date, "10Y"] - data.loc[start_date, "2Y"]
            end_2s10s = data.loc[end_date, "10Y"] - data.loc[end_date, "2Y"]
            spreads["2s10s"] = {
                "start": start_2s10s,
                "end": end_2s10s,
                "change": end_2s10s - start_2s10s,
                "inversion_days": (data["10Y"] - data["2Y"] < 0).sum(),
            }

        # 5s10s spread
        if "5Y" in data.columns and "10Y" in data.columns:
            start_5s10s = data.loc[start_date, "10Y"] - data.loc[start_date, "5Y"]
            end_5s10s = data.loc[end_date, "10Y"] - data.loc[end_date, "5Y"]
            spreads["5s10s"] = {
                "start": start_5s10s,
                "end": end_5s10s,
                "change": end_5s10s - start_5s10s,
                "inversion_days": (data["10Y"] - data["5Y"] < 0).sum(),
            }

        # 10s30s spread
        if "10Y" in data.columns and "30Y" in data.columns:
            start_10s30s = data.loc[start_date, "30Y"] - data.loc[start_date, "10Y"]
            end_10s30s = data.loc[end_date, "30Y"] - data.loc[end_date, "10Y"]
            spreads["10s30s"] = {
                "start": start_10s30s,
                "end": end_10s30s,
                "change": end_10s30s - start_10s30s,
                "inversion_days": (data["30Y"] - data["10Y"] < 0).sum(),
            }

        # Determine curve shape at start and end
        start_shape = self._determine_curve_shape(data.loc[start_date], tenors)
        end_shape = self._determine_curve_shape(data.loc[end_date], tenors)

        return {
            "spreads": spreads,
            "curve_shape": {"start": start_shape, "end": end_shape},
        }

    def _determine_curve_shape(self, yields, tenors):
        """Determine the shape of the yield curve"""
        # Sort tenors by maturity
        tenor_order = ["3M", "2Y", "5Y", "10Y", "30Y"]
        available_tenors = [t for t in tenor_order if t in tenors and t in yields.index]

        if len(available_tenors) < 2:
            return "Insufficient data"

        # Extract yields in order
        ordered_yields = [yields[t] for t in available_tenors]

        # Check if curve is strictly increasing
        is_increasing = all(
            ordered_yields[i] < ordered_yields[i + 1]
            for i in range(len(ordered_yields) - 1)
        )

        # Check if curve is strictly decreasing
        is_decreasing = all(
            ordered_yields[i] > ordered_yields[i + 1]
            for i in range(len(ordered_yields) - 1)
        )

        # Determine curve shape
        if is_increasing:
            return "Normal (upward sloping)"
        elif is_decreasing:
            return "Inverted"
        else:
            # Check for humps
            has_hump = any(
                ordered_yields[i - 1] < ordered_yields[i] > ordered_yields[i + 1]
                for i in range(1, len(ordered_yields) - 1)
            )

            has_dip = any(
                ordered_yields[i - 1] > ordered_yields[i] < ordered_yields[i + 1]
                for i in range(1, len(ordered_yields) - 1)
            )

            if has_hump:
                return "Humped"
            elif has_dip:
                return "Dipped"
            else:
                return "Mixed"

    def _calculate_summary_stats(self, data, tenors):
        """Calculate summary statistics for the cycle"""
        summary = {}

        for tenor in tenors:
            if tenor in data.columns:
                summary[tenor] = {
                    "mean": data[tenor].mean(),
                    "median": data[tenor].median(),
                    "std": data[tenor].std(),
                    "min": data[tenor].min(),
                    "max": data[tenor].max(),
                    "range": data[tenor].max() - data[tenor].min(),
                    "start": data[tenor].iloc[0],
                    "end": data[tenor].iloc[-1],
                    "change": data[tenor].iloc[-1] - data[tenor].iloc[0],
                }

        return summary

    def _calculate_correlations(self, data, tenors):
        """
        Calculate correlations between different tenors

        Args:
            data (DataFrame): Yield data
            tenors (list): List of tenors to analyze

        Returns:
            dict: Correlation data
        """
        if len(tenors) < 2 or len(data) < 3:
            return {}

        # Calculate correlation matrix
        corr_matrix = data[tenors].corr()

        # Convert to dictionary format for easier reporting
        correlations = {}
        for i, tenor1 in enumerate(tenors):
            for tenor2 in tenors[i + 1 :]:  # Only include each pair once
                correlations[f"{tenor1}-{tenor2}"] = corr_matrix.loc[tenor1, tenor2]

        return correlations

    def _test_stationarity(self, data, tenors):
        """Test for stationarity of yield series"""
        stationarity = {}

        for tenor in tenors:
            if tenor in data.columns and len(data) > 20:
                # Run Augmented Dickey-Fuller test
                result = adfuller(data[tenor].dropna())

                stationarity[tenor] = {
                    "adf_statistic": result[0],
                    "p_value": result[1],
                    "is_stationary": result[1]
                    < 0.05,  # p-value < 0.05 indicates stationarity
                    "critical_values": result[4],
                }

        return stationarity

    def decompose_yields(self, data):
        """
        Use PCA to decompose yield curve movements into factors

        Args:
            data (DataFrame): Yield data

        Returns:
            dict: PCA results
        """
        tenors = [col for col in data.columns if col in self.tenors]

        if len(tenors) < 2 or len(data) < 10:
            return {}

        # Prepare data for PCA
        yields_data = data[tenors].dropna()

        if len(yields_data) < 10:
            return {}

        # Standardize the data
        scaler = StandardScaler()
        scaled_yields = scaler.fit_transform(yields_data)

        # Apply PCA
        pca = PCA()
        principal_components = pca.fit_transform(scaled_yields)

        # Prepare results
        component_names = ["Level", "Slope", "Curvature", "Fourth", "Fifth"]

        # Limit to number of tenors (ensure we don't exceed available components)
        num_components = min(len(tenors), len(component_names))
        component_names = component_names[:num_components]

        # Create DataFrame of components - ensure dimensions match
        components_df = pd.DataFrame(
            principal_components[:, :num_components],
            index=yields_data.index,
            columns=component_names,
        )

        return {
            "explained_variance": pca.explained_variance_ratio_.tolist()[
                :num_components
            ],
            "components": components_df,
            "loadings": pd.DataFrame(
                pca.components_[:num_components, :],
                columns=tenors,
                index=component_names,
            ),
        }

    def detect_regime_changes(
        self, data, tenor="10Y", window=30, threshold_multiplier=2.0
    ):
        """
        Detect structural breaks in yield behavior

        Args:
            data (DataFrame): Yield data
            tenor (str): Tenor to analyze
            window (int): Rolling window size
            threshold_multiplier (float): Sensitivity for detection

        Returns:
            list: Dates of detected regime changes
        """
        if tenor not in data.columns or len(data) < window + 10:
            return []

        # Calculate rolling volatility
        volatility = data[tenor].rolling(window=window).std()

        # Compute changes in volatility
        vol_changes = volatility.diff()

        # Identify significant regime changes
        threshold = vol_changes.std() * threshold_multiplier
        regime_changes = vol_changes[abs(vol_changes) > threshold]

        # Convert to list of dates
        regime_dates = regime_changes.index.tolist()

        # Add context for each regime change
        regime_info = []
        for date in regime_dates:
            # Look at the 30-day period following the regime change
            if date + pd.Timedelta(days=30) <= data.index[-1]:
                next_period = data.loc[date : date + pd.Timedelta(days=30), tenor]
                prev_period = data.loc[date - pd.Timedelta(days=30) : date, tenor]

                direction = (
                    "increased"
                    if next_period.mean() > prev_period.mean()
                    else "decreased"
                )
                magnitude = abs(next_period.mean() - prev_period.mean())

                regime_info.append(
                    {
                        "date": date,
                        "direction": direction,
                        "magnitude": magnitude,
                        "tenor": tenor,
                    }
                )

        return regime_info

    def generate_report(self, data, title="Treasury Yield Analysis"):
        """
        Generate a comprehensive report on Treasury yield data

        Args:
            data (DataFrame): Yield data
            title (str): Report title

        Returns:
            str: Formatted report
        """
        # Check if we have data
        if data.empty:
            return "No data available for analysis"

        # Get date range
        start_date = data.index.min().strftime("%Y-%m-%d")
        end_date = data.index.max().strftime("%Y-%m-%d")

        # Run analysis
        try:
            analysis = self.analyze_cycle(data)
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

        # Build report
        report = [f"=== {title} ==="]
        report.append(f"Period: {start_date} to {end_date}")
        report.append(f"Data points: {len(data)}")
        report.append("")

        # Yield changes
        report.append("YIELD CHANGES:")
        for tenor, changes in analysis["yield_changes"].items():
            # Convert to percentage with 2 decimal places
            start_pct = changes["start_yield"] * 100
            end_pct = changes["end_yield"] * 100
            abs_change_pct = changes["absolute_change"] * 100

            report.append(
                f"{tenor}: {start_pct:.2f}% to {end_pct:.2f}% "
                f"(Change: {abs_change_pct:.2f}%, {changes['percentage_change']:.2f}%)"
            )
        report.append("")

        # Curve shape
        report.append("YIELD CURVE CHARACTERISTICS:")
        report.append(
            f"Starting shape: {analysis['curve_characteristics']['curve_shape']['start']}"
        )
        report.append(
            f"Ending shape: {analysis['curve_characteristics']['curve_shape']['end']}"
        )
        report.append("")

        # Key spreads
        report.append("KEY SPREADS:")
        for spread_name, spread_data in analysis["curve_characteristics"][
            "spreads"
        ].items():
            # Convert to percentage with 2 decimal places
            start_pct = spread_data["start"] * 100
            end_pct = spread_data["end"] * 100
            change_pct = spread_data["change"] * 100

            report.append(
                f"{spread_name}: {start_pct:.2f}% to {end_pct:.2f}% "
                f"(Change: {change_pct:.2f}%)"
            )

            # Add inversion information if available
            if "inversion_days" in spread_data:
                report.append(f"  Inverted for {spread_data['inversion_days']} days")
        report.append("")

        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        for tenor, stats in analysis["summary_stats"].items():
            # Convert to percentage with 2 decimal places
            min_pct = stats["min"] * 100
            max_pct = stats["max"] * 100
            range_pct = (stats["max"] - stats["min"]) * 100
            mean_pct = stats["mean"] * 100
            median_pct = stats["median"] * 100
            std_pct = stats["std"] * 100

            report.append(f"{tenor}:")
            report.append(
                f"  Range: {min_pct:.2f}% to {max_pct:.2f}% (Spread: {range_pct:.2f}%)"
            )
            report.append(
                f"  Mean: {mean_pct:.2f}%, Median: {median_pct:.2f}%, Std Dev: {std_pct:.2f}%"
            )
        report.append("")

        # Stationarity tests (if available)
        if "stationarity" in analysis:
            report.append("STATIONARITY TESTS:")
            for tenor, result in analysis["stationarity"].items():
                status = "Stationary" if result["is_stationary"] else "Non-stationary"
                report.append(f"{tenor}: {status} (p-value: {result['p_value']:.4f})")
            report.append("")

        # Correlation (if available)
        if "correlation" in analysis:
            report.append("TENOR CORRELATIONS:")
            for pair, corr in analysis["correlation"].items():
                report.append(f"{pair}: {corr:.4f}")

        return "\n".join(report)

    def plot_yields(self, data, title="Treasury Yields", save_path=None, format="png"):
        """
        Plot Treasury yields over time using Plotly

        Args:
            data (DataFrame): Yield data
            title (str): Plot title
            save_path (str): Path to save the plot (without extension)
            format (str): Output format ('png' or 'html')

        Returns:
            plotly.graph_objects.Figure: The figure object
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Check if all tenors have identical data
        identical_data = True
        first_tenor = None
        for tenor in self.tenors:
            if tenor in data.columns:
                if first_tenor is None:
                    first_tenor = tenor
                elif not data[tenor].equals(data[first_tenor]):
                    identical_data = False
                    break

        # Create a copy of the data for plotting with percentage values
        plot_data = data.copy()

        # Convert yield values to percentages for plotting
        for tenor in self.tenors:
            if tenor in plot_data.columns:
                plot_data[tenor] = plot_data[tenor] * 100

        # Convert spread values to percentages for plotting
        for spread in self.spreads:
            if spread in plot_data.columns:
                plot_data[spread] = plot_data[spread] * 100

        # If all tenors have identical data, create synthetic data for visualization
        if identical_data and first_tenor is not None:
            print(
                "WARNING: All tenors have identical data. Creating synthetic data for visualization."
            )
            base_data = plot_data[first_tenor]
            synthetic_factors = {
                "3M": 0.85,
                "2Y": 0.90,
                "5Y": 0.95,
                "10Y": 1.0,
                "30Y": 1.05,
            }

            for tenor in self.tenors:
                if tenor in plot_data.columns:
                    # Create synthetic data with small offsets to visualize the curve
                    plot_data[f"{tenor}_viz"] = base_data * synthetic_factors[tenor]

                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data[f"{tenor}_viz"],
                            name=f"{tenor} Yield (Synthetic)",
                            line=dict(color=self.colors.get(tenor, None), width=2),
                        ),
                        secondary_y=False,
                    )

            # Add note about synthetic data
            fig.add_annotation(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Note: Synthetic data shown due to identical values across tenors",
                showarrow=False,
                font=dict(size=12, color="red"),
            )
        else:
            # Add yield curves (normal case)
            for tenor in self.tenors:
                if tenor in plot_data.columns and not plot_data[tenor].dropna().empty:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data[tenor],
                            name=f"{tenor} Yield",
                            line=dict(color=self.colors.get(tenor, None), width=2),
                        ),
                        secondary_y=False,
                    )

        # Add spreads on secondary axis
        for spread in self.spreads:
            if spread in plot_data.columns and not plot_data[spread].dropna().empty:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data[spread],
                        name=spread,
                        line=dict(
                            color=self.colors.get(spread, None), width=2, dash="dash"
                        ),
                    ),
                    secondary_y=True,
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            hovermode="x unified",
        )

        # Set y-axes titles with percentage format
        fig.update_yaxes(title_text="Yield (%)", secondary_y=False)
        fig.update_yaxes(title_text="Spread (%)", secondary_y=True)

        # Update hover template to show percentage with 2 decimal places
        fig.update_traces(
            hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>", secondary_y=False
        )
        fig.update_traces(
            hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>", secondary_y=True
        )

        # Save if path is provided
        if save_path:
            # Use the plots directory
            filename = (
                os.path.basename(save_path) if os.path.dirname(save_path) else save_path
            )
            filename = os.path.splitext(filename)[0]  # Remove any extension

            if format.lower() == "html":
                html_path = os.path.join(self.plots_dir, f"{filename}.html")
                fig.write_html(html_path)
                print(f"Interactive plot saved to {html_path}")

            # Always save PNG for static viewing
            png_path = os.path.join(self.plots_dir, f"{filename}.png")
            fig.write_image(png_path, width=1200, height=800)
            print(f"Static plot saved to {png_path}")

        return fig

    def plot_yield_curve(
        self,
        data,
        dates=None,
        title="Treasury Yield Curve",
        save_path=None,
        format="png",
    ):
        """
        Plot the yield curve for specific dates using Plotly

        Args:
            data (DataFrame): Yield data
            dates (list): List of dates to plot
            title (str): Plot title
            save_path (str): Path to save the plot (without extension)
            format (str): Output format ('png' or 'html')

        Returns:
            plotly.graph_objects.Figure: The figure object
        """
        # Get available tenors
        tenors = [col for col in data.columns if col in self.tenors]

        if len(tenors) < 2:
            print("Insufficient tenor data for yield curve plot")
            return None

        # If no dates provided, use first and last date
        if not dates:
            dates = [data.index[0], data.index[-1]]

        # Create figure
        fig = go.Figure()

        # Create a copy of the data for plotting with percentage values
        plot_data = data.copy()

        # Convert yield values to percentages for plotting
        for tenor in tenors:
            plot_data[tenor] = plot_data[tenor] * 100

        # Add yield curves for each date
        for date in dates:
            if date in plot_data.index:
                # Get yields for this date
                yields = plot_data.loc[date, tenors]

                # Convert tenor labels to numeric values for x-axis
                tenor_values = [float(tenor[:-1]) for tenor in tenors]

                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=tenor_values,
                        y=yields,
                        name=date.strftime("%Y-%m-%d"),
                        mode="lines+markers",
                    )
                )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Tenor (Years)",
            yaxis_title="Yield (%)",
            template="plotly_white",
            hovermode="closest",
        )

        # Update hover template to show percentage with 2 decimal places
        fig.update_traces(
            hovertemplate="Tenor: %{x}Y<br>Yield: %{y:.2f}%<extra>%{fullData.name}</extra>"
        )

        # Save if path is provided
        if save_path:
            # Use the plots directory
            filename = (
                os.path.basename(save_path) if os.path.dirname(save_path) else save_path
            )
            filename = os.path.splitext(filename)[0]  # Remove any extension

            if format.lower() == "html":
                html_path = os.path.join(self.plots_dir, f"{filename}.html")
                fig.write_html(html_path)
                print(f"Interactive plot saved to {html_path}")

            # Always save PNG for static viewing
            png_path = os.path.join(self.plots_dir, f"{filename}.png")
            fig.write_image(png_path, width=1200, height=800)
            print(f"Static plot saved to {png_path}")

        return fig

    def plot_pca_components(self, data, save_path=None, format="png"):
        """
        Plot PCA components and loadings using Plotly

        Args:
            data (DataFrame): Yield data
            save_path (str): Path to save the plot (without extension)
            format (str): Output format ('png' or 'html')

        Returns:
            plotly.graph_objects.Figure: The figure object
        """
        # Get PCA results
        pca_results = self.decompose_yields(data)

        if not pca_results:
            print("Unable to perform PCA analysis")
            return None

        components = pca_results["components"]
        loadings = pca_results["loadings"]
        explained_var = pca_results["explained_variance"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("PCA Components Over Time", "PCA Component Loadings"),
            vertical_spacing=0.25,  # Increased spacing
            row_heights=[0.6, 0.4],  # Adjust row heights
        )

        # Plot components over time
        for component in components.columns:
            fig.add_trace(
                go.Scatter(
                    x=components.index,
                    y=components[component],
                    name=component,
                    line=dict(width=2),
                ),
                row=1,
                col=1,
            )

        # Plot loadings as a heatmap instead of bars for better visualization
        heatmap_data = loadings.values

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=loadings.columns,
                y=loadings.index,
                colorscale="RdBu",
                zmid=0,
                text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(title="Loading Value"),
            ),
            row=2,
            col=1,
        )

        # Add explained variance as annotations
        for i, (comp, var) in enumerate(zip(components.columns, explained_var)):
            fig.add_annotation(
                x=0.01,
                y=0.95 - (i * 0.05),
                text=f"{comp}: {var:.2%}",
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="left",
                font=dict(size=12),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=900,  # Increased height
            title_text="PCA Analysis of Treasury Yields",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Component Value", row=1, col=1)
        fig.update_xaxes(title_text="Tenor", row=2, col=1)
        fig.update_yaxes(title_text="Component", row=2, col=1)

        # Save if path is provided
        if save_path:
            # Use the plots directory
            filename = (
                os.path.basename(save_path) if os.path.dirname(save_path) else save_path
            )
            filename = os.path.splitext(filename)[0]  # Remove any extension

            if format.lower() == "html":
                html_path = os.path.join(self.plots_dir, f"{filename}.html")
                fig.write_html(html_path)
                print(f"Interactive plot saved to {html_path}")

            # Always save PNG for static viewing
            png_path = os.path.join(self.plots_dir, f"{filename}.png")
            fig.write_image(png_path, width=1200, height=900)  # Increased height
            print(f"Static plot saved to {png_path}")

        return fig

    def plot_regime_changes(self, data, regime_changes, save_path=None, format="png"):
        """
        Plot yield data with regime change points highlighted using Plotly

        Args:
            data (DataFrame): Yield data
            regime_changes (list): List of regime change dictionaries
            save_path (str): Path to save the plot (without extension)
            format (str): Output format ('png' or 'html')

        Returns:
            plotly.graph_objects.Figure: The figure object
        """
        if not regime_changes:
            print("No regime changes to plot")
            return None

        # Get the tenor used for regime detection
        tenor = regime_changes[0]["tenor"]

        if tenor not in data.columns:
            print(f"Tenor {tenor} not found in data")
            return None

        # Create a copy of the data for plotting with percentage values
        plot_data = data.copy()

        # Convert yield values to percentages for plotting
        plot_data[tenor] = plot_data[tenor] * 100

        # Create figure
        fig = go.Figure()

        # Add yield data
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[tenor],
                name=f"{tenor} Yield",
                line=dict(color=self.colors.get(tenor, None), width=2),
            )
        )

        # Add regime change points
        regime_dates = [regime["date"] for regime in regime_changes]
        regime_values = [plot_data.loc[date, tenor] for date in regime_dates]

        # Create hover text with percentage formatting
        hover_texts = []
        for regime in regime_changes:
            magnitude_pct = regime["magnitude"] * 100
            hover_texts.append(
                f"{regime['date'].strftime('%Y-%m-%d')}: {regime['direction']} by {magnitude_pct:.2f}%"
            )

        fig.add_trace(
            go.Scatter(
                x=regime_dates,
                y=regime_values,
                mode="markers",
                name="Regime Changes",
                marker=dict(
                    color="red", size=10, line=dict(width=2, color="DarkSlateGrey")
                ),
                text=hover_texts,
                hoverinfo="text",
            )
        )

        # Update layout
        fig.update_layout(
            title=f"Treasury Yield ({tenor}) with Regime Changes",
            xaxis_title="Date",
            yaxis_title="Yield (%)",
            template="plotly_white",
            hovermode="closest",
        )

        # Update hover template for the yield line
        fig.update_traces(
            hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>",
            selector=dict(name=f"{tenor} Yield"),
        )

        # Save if path is provided
        if save_path:
            # Use the plots directory
            filename = (
                os.path.basename(save_path) if os.path.dirname(save_path) else save_path
            )
            filename = os.path.splitext(filename)[0]  # Remove any extension

            if format.lower() == "html":
                html_path = os.path.join(self.plots_dir, f"{filename}.html")
                fig.write_html(html_path)
                print(f"Interactive plot saved to {html_path}")

            # Always save PNG for static viewing
            png_path = os.path.join(self.plots_dir, f"{filename}.png")
            fig.write_image(png_path, width=1200, height=800)
            print(f"Static plot saved to {png_path}")

        return fig

    def diagnose_data(self, data):
        """
        Diagnose issues with the yield data

        Args:
            data (DataFrame): Yield data

        Returns:
            str: Diagnostic report
        """
        tenors = [col for col in data.columns if col in self.tenors]

        if len(tenors) < 2:
            return "Insufficient tenor data for diagnosis"

        report = ["DATA DIAGNOSIS:"]

        # Check for identical columns
        identical_cols = []
        for i, tenor1 in enumerate(tenors):
            for tenor2 in tenors[i + 1 :]:
                if data[tenor1].equals(data[tenor2]):
                    identical_cols.append(f"{tenor1} and {tenor2} are identical")

        if identical_cols:
            report.append("WARNING: The following tenor pairs have identical data:")
            report.extend(identical_cols)
            report.append("This suggests an issue with data extraction or processing.")

        # Check for constant values
        constant_cols = []
        for tenor in tenors:
            if data[tenor].nunique() == 1:
                constant_cols.append(
                    f"{tenor} has constant value {data[tenor].iloc[0]}"
                )

        if constant_cols:
            report.append("\nWARNING: The following tenors have constant values:")
            report.extend(constant_cols)

        # Check for missing values
        missing_data = data[tenors].isna().sum()
        if missing_data.sum() > 0:
            report.append("\nMissing values by tenor:")
            for tenor in tenors:
                report.append(f"{tenor}: {missing_data[tenor]} missing values")

        return "\n".join(report)


# Simple test
if __name__ == "__main__":
    import yield_data_fetcher

    # Create fetcher and analyzer
    fetcher = yield_data_fetcher.YieldDataFetcher()
    analyzer = YieldAnalyzer()

    # Try to analyze financial crisis data
    print("Fetching financial crisis data...")
    crisis_data = fetcher.fetch_cycle_data("financial_crisis")

    if not crisis_data.empty:
        # Diagnose the data first
        print("\n" + analyzer.diagnose_data(crisis_data))

        # Print sample of the data to inspect
        print("\nSample data (first 5 rows):")
        print(crisis_data[analyzer.tenors].head())

        # Generate and print report
        report = analyzer.generate_report(crisis_data, "Financial Crisis")
        print(report)

        # Plot the data
        analyzer.plot_yields(
            crisis_data,
            "Financial Crisis Treasury Yields",
            "financial_crisis_yields.html",
        )

        # Try PCA analysis
        try:
            analyzer.plot_pca_components(crisis_data, "financial_crisis_pca.html")
        except Exception as e:
            print(f"Error in PCA analysis: {str(e)}")

        # Detect regime changes
        regime_changes = analyzer.detect_regime_changes(crisis_data)
        print("\nDetected Regime Changes:")
        for regime in regime_changes:
            print(
                f"{regime['date'].strftime('%Y-%m-%d')}: {regime['tenor']} yields {regime['direction']} by {regime['magnitude']:.4f}"
            )

        # Plot regime changes
        analyzer.plot_regime_changes(
            crisis_data, regime_changes, "financial_crisis_regime_changes.html"
        )
    else:
        print("No financial crisis data available for analysis")

    # Try recent data
    print("\nFetching recent data...")
    recent_data = fetcher.fetch_latest_data(days=90)

    if not recent_data.empty:
        # Diagnose the data first
        print("\n" + analyzer.diagnose_data(recent_data))

        # Print sample of the data to inspect
        print("\nSample data (first 5 rows):")
        print(recent_data[analyzer.tenors].head())

        # Generate and print report
        report = analyzer.generate_report(recent_data, "Recent Market")
        print(report)

        # Plot the data
        analyzer.plot_yields(
            recent_data, "Recent Treasury Yields", "recent_yields.html"
        )
    else:
        print("No recent data available for analysis")
