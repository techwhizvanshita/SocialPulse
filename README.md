# SocialPulse: Social Media & E-commerce Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Issues](https://img.shields.io/github/issues/CodesavvySiddharth/SocialPulse)](https://github.com/CodesavvySiddharth/SocialPulse/issues)

## Overview
**SocialPulse** is a sentiment analysis tool that helps businesses monitor and understand public perception of their products and brands across multiple online platforms. Initially using datasets from **Amazon** and **Flipkart**, SocialPulse analyzes user reviews and comments to provide actionable insights.  

The system is designed to be **scalable**, allowing integration of additional e-commerce and social media datasets in the future for broader analysis.

---

## Key Features
- **Multi-Platform Sentiment Analysis**: Analyzes reviews and comments from various platforms to assess product or brand sentiment across different demographics.  
- **Real-Time Tracking & Alerts**: Monitors new reviews or mentions and highlights significant sentiment changes.  
- **Comprehensive Dashboard**: Displays sentiment trends, keyword analysis, and other metrics for strategic decision-making.  
- **Customizable Reports**: Generates detailed reports for specific timeframes, platforms, or sentiment categories.  
- **Future-Ready Dataset Integration**: Easily add more datasets from other e-commerce sites or social media platforms as needed.  

---

## Datasets
- **Current Datasets**:  
  - Amazon Reviews Dataset  
  - Flipkart Reviews Dataset  

- **Future Datasets**:  
  - Designed to integrate additional e-commerce platforms, social media platforms, or any other text-based review sources.  

---

## Built With
- **Python** – Backend development  
- **Streamlit** – Interactive web application  
- **NLP Libraries** – NLTK, spaCy for text processing and sentiment classification  
- **Web Scraping** – BeautifulSoup or Scrapy for data collection (if required)  

---

## Getting Started

### Prerequisites
- Python 3.8 or higher  
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CodesavvySiddharth/SocialPulse.git
   cd SocialPulse
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   streamlit run app.py
4. Open in your browser at:
   ```bash
   http://localhost:8501
