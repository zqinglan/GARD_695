**Treasury Yield Analysis Dashboard**

**Overview**
This project provides a comprehensive analysis of U.S. Treasury yield data, focusing on historical trends, yield curve characteristics, and regime changes. It features an interactive web dashboard that allows users to explore Treasury yield data, analyze historical Fed cutting cycles, and perform custom analyses.

**Features**
Real-time Treasury Yield Data: Fetches and displays the most recent Treasury yield data
Historical Analysis: Analyze yield behavior during past Fed cutting cycles
Custom Date Range Analysis: Select any time period for detailed yield analysis
Yield Curve Visualization: Interactive charts showing yield curves and their evolution
Regime Change Detection: Identifies significant shifts in yield behavior
Principal Component Analysis: Decomposes yield movements into principal components
Percentage-based Display: All yields shown as percentages for easier interpretation

**Components**
The application consists of three main components:
Data Fetcher (yield_data_fetcher.py): Retrieves Treasury yield data from the U.S. Treasury Fiscal Data API
Yield Analyzer (yield_analyzer.py): Analyzes the data to identify patterns and calculate metrics
Web Interface (app.py and templates): Presents the analysis through an interactive dashboard

**Installation**
Prerequisites
Python 3.8+
pip (Python package installer)

**Setup**
1. Clone the repository
2. Install required packages
3. Run the application
4. Access the dashboard at:    _http://localhost:5000/_
