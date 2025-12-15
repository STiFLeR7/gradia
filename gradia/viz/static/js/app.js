// app.js

const state = {
    events: [],
    lastIdx: 0,
    charts: {},
    scenario: null,
    totalEpochs: 0
};

async function fetchEvents() {
    try {
        const res = await fetch('/api/events');
        const allEvents = await res.json();

        // Process new events
        if (allEvents.length > state.lastIdx) {
            const newEvents = allEvents.slice(state.lastIdx);
            newEvents.forEach(handleEvent);
            state.lastIdx = allEvents.length;
        }
    } catch (e) {
        console.error("Failed to fetch events", e);
    }
}

function handleEvent(event) {
    console.log("Event:", event.type, event.data);

    try {
        switch (event.type) {
            case 'train_begin':
                document.getElementById('status-text').innerText = "Training";
                state.scenario = event.data;
                state.totalEpochs = event.data.epochs || 20; // Default fallback
                // Init Progress
                updateProgress(0);

                renderScenarioInfo(event.data);
                initCharts(event.data);
                break;

            case 'epoch_end':
                ensureCharts(); // Robustness
                updateCharts(event.data);
                // document.getElementById('epoch-text').innerText = event.data.epoch; // Removed in favor of bar
                updateMetricsTable(event.data);
                updateProgress(event.data.epoch);
                break;

            case 'system_metrics':
                ensureCharts(); // Robustness
                updateSystemCharts(event.data);
                break;

            case 'train_end':
                document.getElementById('status-text').innerText = "Completed";
                updateProgress(state.totalEpochs); // Ensure full feature
                document.getElementById('status-text').style.color = "var(--success)";
                if (event.data.feature_importance) {
                    updateFeatureImportance(event.data.feature_importance);
                }
                break;
        }
    } catch (err) {
        console.error("Error handling event " + event.type, err);
    }
}

function ensureCharts() {
    if (!state.charts.metric) {
        // Fallback init if train_begin missed
        initCharts({ features: [], target_column: 'Unknown' });
    }
}

function renderScenarioInfo(data) {
    const el = document.getElementById('scenario-details');
    // data.scenario is a string repr, or we use individual fields
    el.innerHTML = `
        Target: <span style="color:white">${data.features ? 'Defined' : 'Auto'}</span><br>
        Features: <span style="color:white">${data.features ? data.features.length : 0}</span>
    `;
}

function initCharts(data) {
    // Metric Chart
    const ctx = document.getElementById('metricChart').getContext('2d');
    state.charts.metric = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Train', data: [], borderColor: '#58a6ff', tension: 0.3 },
                { label: 'Test', data: [], borderColor: '#a371f7', tension: 0.3 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: { y: { grid: { color: '#30363d' } }, x: { grid: { color: '#30363d' } } },
            plugins: { legend: { labels: { color: '#8b949e' } } }
        }
    });

    // System Chart (Gradient Area)
    const ctxSys = document.getElementById('systemChart').getContext('2d');

    // Gradients
    const gradCpu = ctxSys.createLinearGradient(0, 0, 0, 400);
    gradCpu.addColorStop(0, 'rgba(35, 134, 54, 0.5)');
    gradCpu.addColorStop(1, 'rgba(35, 134, 54, 0.0)');

    const gradRam = ctxSys.createLinearGradient(0, 0, 0, 400);
    gradRam.addColorStop(0, 'rgba(247, 129, 102, 0.5)');
    gradRam.addColorStop(1, 'rgba(247, 129, 102, 0.0)');

    state.charts.system = new Chart(ctxSys, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CPU %',
                    data: [],
                    borderColor: '#238636',
                    backgroundColor: gradCpu,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'RAM %',
                    data: [],
                    borderColor: '#f78166',
                    backgroundColor: gradRam,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                y: { min: 0, max: 100, grid: { color: '#30363d' } },
                x: { display: false }
            }
        }
    });

    // FI Chart
    const ctxFi = document.getElementById('fiChart').getContext('2d');
    state.charts.fi = new Chart(ctxFi, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Importance',
                data: [],
                backgroundColor: '#238636',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, indexAxis: 'y',
            scales: { y: { grid: { display: false }, ticks: { color: '#8b949e' } }, x: { grid: { color: '#30363d' } } },
            plugins: { legend: { display: false } }
        }
    });
}

function updateCharts(data) {
    if (!state.charts.metric) return;

    // Check which metrics to plot
    let trainVal = data.train_acc || data.train_mse;
    let testVal = data.test_acc || data.test_mse;

    const chart = state.charts.metric;
    chart.data.labels.push(data.epoch || 1);
    chart.data.datasets[0].data.push(trainVal);
    chart.data.datasets[1].data.push(testVal);

    // Use proper labels
    if (data.train_acc) {
        chart.data.datasets[0].label = "Train Accuracy";
        chart.data.datasets[1].label = "Test Accuracy";
    } else {
        chart.data.datasets[0].label = "Train MSE";
        chart.data.datasets[1].label = "Test MSE";
    }

    chart.update();
}

function updateSystemCharts(data) {
    if (!state.charts.system) return;

    const chart = state.charts.system;
    // Keep max 60 points
    if (chart.data.labels.length > 60) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }

    chart.data.labels.push(""); // timestamp irrelevant for sparkline feel
    chart.data.datasets[0].data.push(data.cpu);
    chart.data.datasets[1].data.push(data.ram);
    chart.update();
}

function updateFeatureImportance(fiMap) {
    if (!state.charts.fi || !fiMap) return;

    const labels = Object.keys(fiMap);
    const values = Object.values(fiMap);

    state.charts.fi.data.labels = labels;
    state.charts.fi.data.datasets[0].data = values;
    state.charts.fi.update();
}

function updateMetricsTable(data) {
    const el = document.getElementById('metrics-table');
    let html = "";

    // Priority metrics first
    const priority = ['train_acc', 'test_acc', 'f1', 'precision', 'recall', 'train_mse', 'test_mse', 'mae', 'r2'];

    // Process priority first
    for (const key of priority) {
        if (data[key] !== undefined) {
            html += renderMetricRow(key, data[key]);
        }
    }

    // Rest
    for (const [k, v] of Object.entries(data)) {
        if (typeof v === 'number' && k !== 'epoch' && !priority.includes(k)) {
            html += renderMetricRow(k, v);
        }
    }
    el.innerHTML = html;
}

function renderMetricRow(k, v) {
    // Pretty names
    const names = {
        'train_acc': 'Train Accuracy', 'test_acc': 'Test Accuracy',
        'train_mse': 'Train MSE', 'test_mse': 'Test MSE',
        'f1': 'F1 Score', 'precision': 'Precision', 'recall': 'Recall',
        'mae': 'MAE', 'r2': 'R² Score'
    };

    const label = names[k] || k;
    const color = k.startsWith('test') || k === 'f1' ? 'var(--accent)' : 'var(--text-secondary)';

    let valStr = String(v);
    if (typeof v === 'number') {
        valStr = v.toFixed(4);
    } else if (v === null || v === undefined) {
        valStr = "-";
    }

    return `<div style="display:flex; justify-content:space-between; margin-bottom:8px; border-bottom:1px solid #202020; padding-bottom:4px">
        <span style="color:${color}; font-size:0.85rem">${label}</span>
        <span style="font-family:monospace; font-weight:600">${valStr}</span>
    </div>`;
}

function updateProgress(current) {
    const total = state.totalEpochs;
    const pct = Math.min((current / total) * 100, 100);

    document.getElementById('progress-fill').style.width = `${pct}%`;
    document.getElementById('progress-text').innerText = `${current} / ${total} Epochs`;
}

// ... (polling)
let pollInterval = setInterval(fetchEvents, 500);
fetchEvents();

/* --- New Actions --- */

function downloadReport(type) {
    if (type === 'json') {
        window.open('/api/report/json', '_blank');
    } else {
        window.open('/api/report/pdf', '_blank');
    }
}

async function runEvaluation() {
    const btn = document.querySelector('button[onclick="runEvaluation()"]');
    btn.innerText = "Running...";
    btn.disabled = true;

    try {
        const res = await fetch('/api/evaluate', { method: 'POST' });
        const data = await res.json();

        if (data.confusion_matrix) {
            renderConfusionMatrix(data.confusion_matrix, data.classes);
        } else {
            alert("Evaluation passed, but only classification supports Matrix currently.");
        }
    } catch (e) {
        alert("Evaluation failed: " + e);
    } finally {
        btn.innerText = "Run Evaluation";
        btn.disabled = false;
    }
}

function renderConfusionMatrix(matrix, classes) {
    document.getElementById('eval-card').style.display = 'block';
    const container = document.getElementById('cm-container');

    // Simple HTML heatmap
    let html = '<table style="border-collapse: collapse; font-size: 0.8rem;">';

    // Header
    html += '<tr><td></td>';
    classes.forEach(c => html += `<td style="padding:5px; font-weight:bold; color:var(--text-secondary)">${c}</td>`);
    html += '</tr>';

    // Find max for color scaling
    let maxVal = 0;
    matrix.forEach(row => row.forEach(v => maxVal = Math.max(maxVal, v)));

    matrix.forEach((row, i) => {
        html += `<tr><td style="padding:5px; font-weight:bold; color:var(--text-secondary)">${classes[i]}</td>`;
        row.forEach(val => {
            const intensity = maxVal > 0 ? val / maxVal : 0;
            // Green intensity
            const bg = `rgba(16, 163, 127, ${intensity * 0.8 + 0.1})`;
            html += `<td style="background:${bg}; padding:10px 15px; text-align:center; border:1px solid #333; color:white;">${val}</td>`;
        });
        html += '</tr>';
    });

    html += '</table>';
    container.innerHTML = html;

    // Scroll to it
    document.getElementById('eval-card').scrollIntoView({ behavior: 'smooth' });
}
