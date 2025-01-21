document.addEventListener('DOMContentLoaded', function() {
  const articleText = document.getElementById('articleText');
  const analyzeButton = document.getElementById('analyzeButton');
  const loadingState = document.getElementById('loadingState');
  const resultContainer = document.getElementById('resultContainer');
  const errorMessage = document.getElementById('errorMessage');
  const apiStatus = document.getElementById('apiStatus');

  // Check API status on load
  checkApiStatus();

  analyzeButton.addEventListener('click', analyzeText);

  async function checkApiStatus() {
      try {
          const response = await fetch('http://127.0.0.1:5000/health');
          const data = await response.json();
          
          if (data.status === 'healthy') {
              apiStatus.className = 'mb-4 p-2 rounded text-sm bg-green-100 text-green-800';
              apiStatus.textContent = 'API Connected';
          } else {
              apiStatus.className = 'mb-4 p-2 rounded text-sm bg-yellow-100 text-yellow-800';
              apiStatus.textContent = '⚠ API Status: Degraded';
          }
      } catch (error) {
          apiStatus.className = 'mb-4 p-2 rounded text-sm bg-red-100 text-red-800';
          apiStatus.textContent = '✕ API Unavailable';
      }
  }

  async function analyzeText() {
      const text = articleText.value.trim();
      
      if (!text) {
          showError('Please enter some text to analyze');
          return;
      }

      try {
          showLoading(true);
          
          const response = await fetch('http://127.0.0.1:5000/predict', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({ text }),
          });

          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          displayResults(data);
      } catch (error) {
          showError('Error analyzing text: ' + error.message);
      } finally {
          showLoading(false);
      }
  }

  function displayResults(data) {
      const isFake = data.prediction.is_fake;
      const mainResult = document.getElementById('mainResult');
      const probabilities = document.getElementById('probabilities');
      const sentiment = document.getElementById('sentiment');
      const textStats = document.getElementById('textStats');

      // Main Result
      mainResult.innerHTML = `
          
          <div class="text-xl font-bold ${isFake ? 'text-red-600' : 'text-green-600'} mb-2">
              ${isFake ? 'Likely Fake News' : 'Likely Real News'}
          </div>
          <div class="confidence-bar ${isFake ? 'fake-news' : 'real-news'}">
              <div class="confidence-fill" style="width: ${data.prediction.confidence * 100}%"></div>
          </div>
          <div class="text-sm text-gray-600 mt-1">
              ${(data.prediction.confidence * 100).toFixed(1)}% Confidence
          </div>
      `;

      // Probabilities
      probabilities.innerHTML = `
          <div class="mb-1">
              <span class="text-red-600">Fake: ${(data.probability.fake * 100).toFixed(1)}%</span>
          </div>
          <div>
              <span class="text-green-600">Real: ${(data.probability.real * 100).toFixed(1)}%</span>
          </div>
      `;

      // Sentiment
      sentiment.innerHTML = `
          <div class="mb-1">
              Polarity: ${data.sentiment.polarity.toFixed(2)}
          </div>
          <div>
              Subjectivity: ${data.sentiment.subjectivity.toFixed(2)}
          </div>
      `;

      // Text Stats
      textStats.innerHTML = `
          <div>Words: ${data.text_analysis.word_count}</div>
          <div>Unique Words: ${data.text_analysis.unique_word_count}</div>
          <div>Diversity: ${(data.text_analysis.unique_word_ratio * 100).toFixed(1)}%</div>
          <div>Avg Length: ${data.text_analysis.avg_word_length.toFixed(1)}</div>
      `;

      resultContainer.classList.remove('hidden');
  }

  function getSentimentEmoji(polarity) {
      if (polarity > 0.3) return '😊';
      if (polarity < -0.3) return '😞';
      return '😐';
  }

  function showLoading(show) {
      loadingState.classList.toggle('hidden', !show);
      resultContainer.classList.toggle('hidden', show);
      errorMessage.classList.add('hidden');
  }

  function showError(message) {
      errorMessage.textContent = message;
      errorMessage.classList.remove('hidden');
  }
});