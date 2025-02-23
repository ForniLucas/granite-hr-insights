# Granite HR Insights  

**Granite HR Insights** is an AI-driven application designed to empower HR managers with actionable employee performance and satisfaction insights. By integrating advanced technologies like IBM granite models,  and data visualization tools, it provides concise, bias-checked reports to enhance decision-making and foster a healthier workplace.

---

## **Features**  

1. **Flexible Data Input**  
   - Upload a CSV file with employee data or use the built-in example dataset.  

2. **Individual Employee Reports**  
   - Generates bullet-point summaries including strengths, areas for improvement, and KPIs using Granite Instruct
   - Detects potential social bias in feedback using Granite Guardian.  

3. **Department-Level Insights**  
   - Delivers aggregated reports with team trends and actionable recommendations.  
   - Visualizes key metrics like satisfaction and performance distributions.  

4. **Bias Detection**  
   - Analyzes supervisor feedback, peer reviews, and Q&A responses for fairness and objectivity.  

5. **Interactive Visualizations**  
   - Includes bar charts, scatter plots, and line graphs for intuitive data exploration.  

---

## **Technologies Used**  

- **Granite-3-8B-Instruct - IBM WatsonX AI**  
   - Powers concise and professional report generation.  

- **Granite Guardian (3.1-2B)**  
   - Identifies social bias in textual feedback with probabilistic scoring.  

- **Streamlit**  
   - Provides an interactive web interface for data upload and report viewing.  

- **Pandas & Matplotlib/Seaborn**  
   - Handles data processing and generates insightful visualizations.  

- **Transformers (Hugging Face)**  
   - Supports model loading and tokenization for Granite Guardian.  

---

## **How It Works**  

1. **Data Input**  
   - Users upload a CSV file or rely on a default example dataset with employee metrics.  

2. **Report Generation**  
   - Select an employee or department to generate tailored reports.  
   - AI processes feedback, Q&A, and numerical data to produce concise summaries.  

3. **Bias Analysis**  
   - Feedback is screened for social bias, with results displayed alongside probabilities.  

4. **Visualization**  
   - Interactive charts highlight individual and departmental trends for quick insights.  

---

## **Synthetic Dataset**  

Due to the scarcity of datasets with textual responses—owing to their sensitive nature—we created a small synthetic dataset for testing and development. Its fields are:  

- **EmployeeID**: Unique identifier (e.g., "E001").  
- **Department**: Employee’s department (e.g., "IT").  
- **PerformanceRating**: Score from 1–5 (e.g., 4).  
- **SatisfactionScore**: Score from 1–5 (e.g., 3).  
- **SupervisorFeedback**: Textual evaluation (e.g., "Demonstrates great technical ability...").  
- **PeerFeedback**: Peer review text (e.g., "Very collaborative...").  
- **Attrition**: Binary flag (0 = retained, 1 = left).  
- **HireDate**: Employment start date (e.g., "2018-06-15").  
- **YearsOfService**: Years employed (e.g., 6).  
- **EducationLevel**: Degree attained (e.g., "Bachelor's Degree").  
- **JobRole**: Position (e.g., "Software Engineer").  
- **EmployeeQnA**: Q&A responses (e.g., "Q: What motivates you? A: Solving complex problems...").  

This dataset was inspired by structures from publicly available sources (see "Dataset References" below).  

---

## **Usage**  

1. Open the app in your browser.  
2. Upload a CSV file with employee data or use the example dataset.  
3. Select an employee or department:  
   - **Individual Report**: View a bias-checked summary with visualizations.  
   - **Department Report**: Explore aggregated insights and trends.  
4. Interpret results and act on recommendations.  

---

## **Future Enhancements**  

- Employee feedback collection.  
- Add predictive analytics for attrition risk, with plans to expand this feature by leveraging granite-timeseries models for enhanced time-series forecasting and trend analysis.  
- Expand bias detection to cover additional risk types.  

---

## **Contributing**  

Contributions are welcome! Fork the repository, make your changes, and submit a pull request.  

---

## **Dataset References**  

Our synthetic dataset draws inspiration from the following sources:  

- Pavan Subhasht, "IBM HR Analytics Attrition Dataset," *Kaggle*, [https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data).  
- Mexwell, "Employee Performance and Productivity Data," *Kaggle*, [https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data).  
- Sanjan Chaudhari, "Employee Performance for HR Analytics," *Kaggle*, [https://www.kaggle.com/datasets/sanjanchaudhari/employees-performance-for-hr-analytics](https://www.kaggle.com/datasets/sanjanchaudhari/employees-performance-for-hr-analytics).  
- City of Tempe, "Employee Survey," *data.gov*, [https://catalog.data.gov/dataset/](https://catalog.data.gov/dataset/).  
- A. Patil et al., "Employee Attrition Prediction Using Machine Learning," *arXiv*, 2017, [https://arxiv.org/pdf/1712.00991](https://arxiv.org/pdf/1712.00991).  
- Pierce County, "Employee Data," *open.piercecountywa.gov*, [https://open.piercecountywa.gov/w/gw2z-y7be/](https://open.piercecountywa.gov/w/gw2z-y7be/).  

---

## **License**  

This project is licensed under the MIT License.  
