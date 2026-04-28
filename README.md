# Subscription Value Brain

An end-to-end customer value intelligence system that combines **churn prediction, customer lifetime value estimation, uplift modeling, and campaign targeting** to help subscription businesses identify which customers should receive retention offers.

This project demonstrates how data science can support business decisions by answering a key question:

> Which customers are most likely to churn, are valuable enough to save, and are actually likely to respond to a retention offer?

---

## Overview

Subscription-based businesses such as SaaS platforms, streaming services, fitness apps, and digital products often spend heavily on retention campaigns. However, not every customer should receive a discount or marketing offer.

Some customers may stay without an offer.  
Some customers may leave regardless of the offer.  
Some customers may respond positively to the offer.  
Some customers may even react negatively to unnecessary outreach.

This project builds a production-style machine learning system that helps businesses prioritize retention campaigns using a unified customer targeting score.

The system combines:

- **Churn Prediction** to estimate who is likely to leave
- **Customer Lifetime Value** to estimate how valuable each customer is
- **Uplift Modeling** to estimate whether marketing will actually help
- **Targeting Score** to rank customers by expected retained value
- **Streamlit Dashboard** to simulate business campaign decisions

---

## Problem Statement

Retention campaigns are often expensive and poorly targeted. A company may waste money by offering discounts to customers who would have stayed anyway, or by targeting customers who are unlikely to respond.

This project answers four business questions:

1. Who is likely to churn?
2. How valuable is each customer?
3. Will a retention offer actually change the outcome?
4. Which customers should be targeted first under a limited budget?

---

## Key Features

- End-to-end customer retention modeling system
- Churn prediction using Logistic Regression and XGBoost
- Customer Lifetime Value estimation using RFM-based features
- Uplift modeling using a two-model treatment/control approach
- Customer segmentation into actionable marketing groups
- Unified targeting score for campaign prioritization
- Budget-based campaign simulation
- Downloadable target customer list
- Streamlit dashboard for business users
- Methodology page explaining modeling decisions
- Clean project structure with notebooks, source code, and processed data

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Modeling | Logistic Regression, Random Forest, XGBoost |
| Evaluation | ROC-AUC, Precision-Recall, Precision@K |
| App Framework | Streamlit |
| Model Persistence | Joblib |
| Development | Jupyter Notebook |
| Deployment Ready | Streamlit-compatible app structure |

---

## System Workflow

    Customer Data
        ↓
    Data Cleaning and Feature Engineering
        ↓
    Churn Prediction Model
        ↓
    Customer Lifetime Value Model
        ↓
    Uplift Modeling
        ↓
    Unified Targeting Score
        ↓
    Customer Segmentation
        ↓
    Campaign Simulation Dashboard
        ↓
    Downloadable Target List

## Modeling Approach
**1. Churn Prediction**

The churn model predicts the probability that a customer will cancel or stop using the service.

Models used:

Logistic Regression
XGBoost Classifier

Evaluation metrics:

ROC-AUC
Precision-Recall
Precision@K

Output:

p_churn = probability that a customer is likely to churn

**2. Customer Lifetime Value Estimation**

The CLV model estimates the expected future value of each customer.

Feature approach:

Recency
Frequency
Monetary value
Subscription behavior
Customer engagement patterns

Model used:

Random Forest Regressor

Additional technique:

Log transformation on monetary values to reduce the effect of skewed revenue distribution

Output:

clv = expected future customer value

**3. Uplift Modeling**

The uplift model estimates whether a retention offer is likely to make a positive difference.

This project uses a two-model approach:

Treatment Model → Probability of retention with offer
Control Model   → Probability of retention without offer

Uplift is calculated as:

uplift = P(retention | treatment) - P(retention | control)

Model used:

Random Forest Classifier

Output:

uplift = estimated incremental impact of marketing treatment

**4. Customer Segmentation**

Customers are grouped into business-friendly uplift segments:

Segment	Meaning
Persuadable	Customers likely to respond positively to an offer
Sure Thing	Customers likely to stay even without an offer
Lost Cause	Customers unlikely to stay even with an offer
Do-Not-Disturb	Customers who may react negatively to unnecessary targeting

This helps marketing teams avoid wasting budget and focus on customers where intervention can create real business value.

**5. Unified Targeting Score**

The final targeting score combines churn risk, customer value, and positive uplift.

target_score = p_churn × clv × uplift_positive

This score ranks customers by expected retained revenue.

A high score means the customer is:

Likely to churn
Valuable to the business
Likely to respond positively to a retention offer
## Streamlit Dashboard

The dashboard allows users to explore retention targeting decisions interactively.

Dashboard features include:

Select top X% of customers to target
Simulate campaign budget constraints
View expected retained value
Analyze uplift segment distribution
Compare customer groups
Download a prioritized target list
Read the methodology behind the modeling approach
Example Business Scenario

A subscription company has 100,000 customers but only enough budget to send offers to 10,000 customers.

Instead of targeting customers only based on churn risk, this system helps identify customers who are both valuable and likely to respond to the offer.

## Example decision logic:

Do not target every high-churn customer.
Target customers with high churn risk, high CLV, and positive uplift.

This helps reduce wasted marketing spend and improves campaign return on investment.

## Project Structure
subscription-value-brain/
│
├── data/
│   └── processed/
│       └── Processed datasets used for modeling and dashboarding
│
├── notebooks/
│   └── Exploratory analysis and model development notebooks
│
├── src/
│   └── Source code for modeling, scoring, and dashboard logic
│
├── requirements.txt
└── README.md

## Getting Started
**1. Clone the Repository**
   git clone https://github.com/Shreevikas-BJ/subscription-value-brain.git
cd subscription-value-brain

**2. Create a Virtual Environment**
    python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate

**3. Install Dependencies**
    pip install -r requirements.txt
    
## Run the Project

If the project includes a Streamlit app file, run:

streamlit run src/app.py

If your Streamlit file has a different name or location, update the command accordingly.

## Model Outputs

The system produces customer-level outputs such as:

customer_id
p_churn
clv
uplift
uplift_segment
target_score
target_rank

These outputs can be used by marketing, retention, or customer success teams to prioritize outreach.

## Evaluation Metrics
**Churn Model**
ROC-AUC
Precision
Recall
Precision@K
Precision-Recall Curve
**CLV Model**
MAE
RMSE
R² Score
**Uplift Model**
Uplift segment distribution
Treatment vs control response comparison
Incremental targeting value
## Why This Project Matters

This project goes beyond a simple churn prediction model.

A normal churn model only answers:

Who might leave?

This project answers a more business-useful question:

Who should we spend money trying to retain?

That distinction is important because business teams need models that support decisions, not just predictions.

## Business Value

This system can help subscription businesses:

Reduce unnecessary discount spending
Improve retention campaign targeting
Prioritize high-value customers
Avoid targeting customers who would stay anyway
Identify customers most likely to respond to offers
Improve customer lifetime value
Support data-driven marketing decisions
## Key Learnings

This project demonstrates:

End-to-end data science workflow design
Business-driven machine learning
Customer churn modeling
CLV estimation
Uplift modeling
Campaign simulation
Model evaluation beyond accuracy
Translating ML outputs into business decisions
Building stakeholder-friendly dashboards

## Future Improvements
Add real-time scoring API using FastAPI
Add automated retraining pipeline
Add MLflow experiment tracking
Add model monitoring and drift detection
Add SHAP-based model explainability
Add campaign ROI calculator
Add customer-level recommendation reasons
Add A/B testing simulation
Add Docker support
Add cloud deployment
Add database integration with Snowflake or PostgreSQL
## Repository Description

End-to-end customer value engine combining churn prediction, CLV estimation, and uplift modeling to identify which customers should receive retention offers.

## Author

Shreevikas Bangalore Jagadish
Graduate Student, Information Technology and Management
Illinois Institute of Technology

GitHub: Shreevikas-BJ
LinkedIn: shreevikasbj
Portfolio: datascienceportfol.io/shreevikasbj
## Repository

https://github.com/Shreevikas-BJ/subscription-value-brain 
