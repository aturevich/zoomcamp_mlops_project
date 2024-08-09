function updateRecentPredictions() {
    fetch('/monitoring/recent_predictions')
        .then(response => response.json())
        .then(data => {
            Plotly.newPlot('recent-predictions', [{
                type: 'scatter',
                mode: 'markers',
                x: data.map(d => d.longitude),
                y: data.map(d => d.latitude),
                text: data.map(d => `Magnitude: ${d.magnitude}<br>Depth: ${d.depth}`),
                marker: {
                    size: data.map(d => d.magnitude * 5),
                    color: data.map(d => d.depth),
                    colorscale: 'Viridis'
                }
            }], {
                title: 'Recent Predictions',
                xaxis: {title: 'Longitude'},
                yaxis: {title: 'Latitude'}
            });
        });
}

function updateDriftResults() {
    fetch('/monitoring/drift_results')
        .then(response => response.json())
        .then(data => {
            Plotly.newPlot('drift-results', [{
                type: 'scatter',
                mode: 'lines+markers',
                x: data.map(d => d.timestamp),
                y: data.map(d => d.drift_score),
                name: 'Drift Score'
            }], {
                title: 'Data Drift Over Time',
                xaxis: {title: 'Timestamp'},
                yaxis: {title: 'Drift Score'}
            });
        });
}

// Update every 60 seconds
setInterval(updateRecentPredictions, 60000);
setInterval(updateDriftResults, 60000);

// Initial update
updateRecentPredictions();
updateDriftResults();
