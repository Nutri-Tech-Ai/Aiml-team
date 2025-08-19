class FertilizerPredictionAPI {
    constructor(baseURL = 'http://localhost:5000') {
        this.baseURL = baseURL;
    }

    async predict(formData) {
        try {
            const response = await fetch(`${this.baseURL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Prediction failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error', model_loaded: false };
        }
    }
}

// Initialize API client
const api = new FertilizerPredictionAPI();

// Form handling
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        showLoading();
        
        try {
            // Collect form data
            const formData = {
                District_Name: document.getElementById('District_Name').value,
                Soil_color: document.getElementById('Soil_color').value,
                Nitrogen: parseFloat(document.getElementById('Nitrogen').value),
                Phosphorus: parseFloat(document.getElementById('Phosphorus').value),
                Potassium: parseFloat(document.getElementById('Potassium').value),
                pH: parseFloat(document.getElementById('pH').value),
                Rainfall: parseFloat(document.getElementById('Rainfall').value),
                Temperature: parseFloat(document.getElementById('Temperature').value),
                Crop: document.getElementById('Crop').value
            };

            // Make prediction
            const result = await api.predict(formData);
            
            // Display results
            displayResults(result);
            
        } catch (error) {
            displayError(error.message);
        } finally {
            hideLoading();
        }
    });

    function showLoading() {
        loadingDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
    }

    function hideLoading() {
        loadingDiv.style.display = 'none';
    }

    function displayResults(result) {
        resultsDiv.innerHTML = `
            <div class="result-success">
                <h3>🌾 Prediction Results</h3>
                <div class="fertilizer-recommendation">
                    <strong>Recommended Fertilizer:</strong> ${result.fertilizer}
                </div>
                <div class="confidence-score">
                    <strong>Confidence:</strong> ${result.confidence}%
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence}%"></div>
                </div>
            </div>
        `;
        resultsDiv.style.display = 'block';
    }

    function displayError(message) {
        resultsDiv.innerHTML = `
            <div class="result-error">
                <h3>❌ Prediction Error</h3>
                <p>${message}</p>
            </div>
        `;
        resultsDiv.style.display = 'block';
    }

    // Check API health on page load
    api.healthCheck().then(health => {
        if (!health.model_loaded) {
            console.warn('Model not loaded on server');
        }
    });
});