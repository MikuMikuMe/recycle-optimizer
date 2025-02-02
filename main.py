Creating a comprehensive project like "recycle-optimizer" involves several components, including data collection, preprocessing, model training, and deployment. Below is a simplified outline and initial implementation of such a tool in Python. This version focuses on simulating the process since building a full-scale implementation requires much more depth, especially on data acquisition and detailed domain-specific algorithms.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

# Setup logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Simulated data: Normally, you would load a dataset relevant to recycling optimization
# For this example, we'll create a random dataset
def generate_mock_data(num_samples=1000):
    # Features: material_type (0: plastic, 1: metal, 2: paper), contamination_level (0-10 scale), weight_kg
    np.random.seed(42)
    data = {
        'material_type': np.random.choice([0, 1, 2], size=num_samples),
        'contamination_level': np.random.randint(0, 11, size=num_samples),
        'weight_kg': np.random.uniform(0.1, 5.0, size=num_samples),
        'recycle': np.random.choice([0, 1], size=num_samples)  # Target: 0: avoid recycling, 1: recycle
    }
    return pd.DataFrame(data)

# Load and prepare the data
def load_data():
    logging.info("Loading and preparing data...")
    # In a real scenario, you'd load actual data related to waste management and recycling
    data = generate_mock_data()
    X = data[['material_type', 'contamination_level', 'weight_kg']]
    y = data['recycle']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    logging.info("Training the model...")
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error in training the model: %s", e)
        return None

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    if model is None:
        logging.error("Model is not trained. Evaluation is skipped.")
        return
    logging.info("Evaluating the model...")
    try:
        predictions = model.predict(X_test)
        logging.info("\n" + classification_report(y_test, predictions))
        logging.info("Accuracy: %.2f%%" % (accuracy_score(y_test, predictions) * 100))
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)

# Run the main application
def main():
    logging.info("Recycle Optimizer Application Started")
    try:
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        logging.error("An error occurred in the main application: %s", e)

if __name__ == "__main__":
    main()
```

### Explanation & Considerations

1. **Data Generation**: Since there's no accessible real-world dataset for this example, the program generates synthetic data. This should be replaced by actual data for a full implementation.

2. **Model**: A simple Random Forest Classifier is used for classification (recyclable or not). This can be replaced or enhanced with more complex algorithms or models tailored to specific waste management scenarios.

3. **Logging and Error Handling**: Comprehensive logging and error handling are included to ensure that any issues during data loading, training, or evaluation are logged properly.

4. **Scalability and Real-world Application**: For real-world applications, this program would need integration with IoT devices or waste management systems for data collection, a much larger dataset accompanied by proper preprocessing, and possibly a web/interface for real-time optimization suggestions.

5. **Modular Design**: The code structure is designed for clarity and modularity, making it easier to extend or replace components as the project grows.

This code serves as a basic framework. To move beyond this, consider enhancing data acquisition, refining feature engineering, and working closely with recycling facility experts to tailor models for specific operational contexts.