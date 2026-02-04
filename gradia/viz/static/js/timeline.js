// timeline.js - Learning Timeline v2.0

const timelineState = {
    summaries: [],
    insights: null,
    currentSample: null,
    charts: {},
    pollInterval: null,
    lastSummaryCount: 0
};

// ============================================================================
// Initialization
// ============================================================================

async function initTimeline() {
    initStabilityChart();
    startPolling();
    await fetchInitialData();
}

function startPolling() {
    timelineState.pollInterval = setInterval(async () => {
        await fetchSummaries();
        await fetchInsights();
    }, 1000);
}

async function fetchInitialData() {
    await fetchSummaries();
    await fetchInsights();
}

// ============================================================================
// Data Fetching
// ============================================================================

async function fetchSummaries() {
    try {
        const res = await fetch('/api/timeline/summaries');
        if (!res.ok) return;
        
        const summaries = await res.json();
        
        if (summaries.length > timelineState.lastSummaryCount) {
            timelineState.summaries = summaries;
            timelineState.lastSummaryCount = summaries.length;
            updateTimelineOverview();
            updateTrainingContext();
        }
    } catch (e) {
        console.error('Failed to fetch summaries:', e);
    }
}

async function fetchInsights() {
    try {
        const res = await fetch('/api/timeline/insights');
        if (!res.ok) return;
        
        const insights = await res.json();
        timelineState.insights = insights;
        
        updateInstabilityPanel();
        updateStabilityChart();
    } catch (e) {
        console.error('Failed to fetch insights:', e);
    }
}

async function fetchSampleTimeline(sampleId) {
    try {
        const res = await fetch(`/api/timeline/sample/${sampleId}`);
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.error('Failed to fetch sample timeline:', e);
        return null;
    }
}

// ============================================================================
// Timeline Overview (Block A)
// ============================================================================

function updateTimelineOverview() {
    const container = document.getElementById('timeline-overview');
    const insights = timelineState.insights;
    const summaries = timelineState.summaries;
    
    if (!insights || !summaries.length) {
        container.innerHTML = '<div class="timeline-loading"><span>Waiting for training data...</span></div>';
        return;
    }
    
    const trackedSamples = insights.tracked_samples || [];
    const maxEpoch = Math.max(...summaries.map(s => s.epoch), 1);
    
    // Build sample stability map from insights
    const stabilityMap = {};
    
    // Process all stability data from insights
    for (const item of insights.top_flipping || []) {
        stabilityMap[item.sample_id] = item.stability_class;
    }
    for (const item of insights.late_learners || []) {
        stabilityMap[item.sample_id] = 'late_learner';
    }
    for (const item of insights.never_correct || []) {
        stabilityMap[item.sample_id] = 'stable_wrong';
    }
    
    // Build HTML
    let html = '';
    
    // Epoch axis
    html += '<div class="epoch-axis">';
    for (let e = 1; e <= maxEpoch; e++) {
        html += `<span class="epoch-label">${e}</span>`;
    }
    html += '</div>';
    
    html += '<div class="timeline-grid">';
    
    // Limit to first 50 samples for performance
    const displaySamples = trackedSamples.slice(0, 50);
    
    for (const sampleId of displaySamples) {
        const stability = stabilityMap[sampleId] || 'unknown';
        
        html += `<div class="timeline-row">`;
        html += `<span class="timeline-sample-id" onclick="openSampleInspector(${sampleId})">#${sampleId}</span>`;
        html += `<div class="timeline-epochs">`;
        
        // Create epoch cells (simplified - full implementation would fetch per-sample events)
        for (let e = 1; e <= maxEpoch; e++) {
            const cellClass = getStabilityCellClass(stability, e, maxEpoch);
            html += `<div class="epoch-cell ${cellClass}" 
                         onclick="openSampleInspector(${sampleId})"
                         title="Sample ${sampleId}, Epoch ${e}"></div>`;
        }
        
        html += '</div></div>';
    }
    
    html += '</div>';
    
    if (trackedSamples.length > 50) {
        html += `<div style="text-align: center; padding: 10px; color: var(--text-secondary); font-size: 0.8rem;">
            Showing 50 of ${trackedSamples.length} tracked samples
        </div>`;
    }
    
    container.innerHTML = html;
}

function getStabilityCellClass(stability, epoch, maxEpoch) {
    // Simplified visualization based on stability class
    switch (stability) {
        case 'stable_correct':
            return 'correct';
        case 'stable_wrong':
            return 'wrong';
        case 'unstable':
            // Alternate for visual effect
            return epoch % 2 === 0 ? 'correct' : 'wrong';
        case 'late_learner':
            // Show transition
            const transitionPoint = Math.floor(maxEpoch * 0.6);
            return epoch >= transitionPoint ? 'correct' : 'wrong';
        default:
            return '';
    }
}

// ============================================================================
// Training Context (Block D)
// ============================================================================

function updateTrainingContext() {
    const summaries = timelineState.summaries;
    const insights = timelineState.insights;
    
    if (summaries.length > 0) {
        const latest = summaries[summaries.length - 1];
        document.getElementById('current-epoch').textContent = latest.epoch;
        document.getElementById('training-status').textContent = 'Training';
        document.getElementById('training-status').style.color = 'var(--accent)';
    }
    
    if (insights) {
        document.getElementById('tracked-count').textContent = insights.total_tracked || 0;
    }
}

// ============================================================================
// Instability Panel (Block C)
// ============================================================================

function updateInstabilityPanel() {
    const insights = timelineState.insights;
    if (!insights) return;
    
    // Top Flipping
    const flippingList = document.getElementById('flipping-list');
    if (insights.top_flipping && insights.top_flipping.length > 0) {
        flippingList.innerHTML = insights.top_flipping.slice(0, 5).map(item => `
            <div class="sample-item" onclick="openSampleInspector(${item.sample_id})">
                <span class="sample-id">#${item.sample_id}</span>
                <span class="sample-meta">Label: ${item.true_label}</span>
                <span class="flip-badge">${item.flip_count} flips</span>
            </div>
        `).join('');
    } else {
        flippingList.innerHTML = '<span class="empty-state">No flipping samples detected</span>';
    }
    
    // Late Learners
    const lateList = document.getElementById('late-learners-list');
    if (insights.late_learners && insights.late_learners.length > 0) {
        lateList.innerHTML = insights.late_learners.slice(0, 5).map(item => `
            <div class="sample-item" onclick="openSampleInspector(${item.sample_id})">
                <span class="sample-id">#${item.sample_id}</span>
                <span class="sample-meta">First correct: Epoch ${item.first_correct_epoch}</span>
            </div>
        `).join('');
    } else {
        lateList.innerHTML = '<span class="empty-state">No late learners detected</span>';
    }
    
    // Never Correct
    const neverList = document.getElementById('never-correct-list');
    if (insights.never_correct && insights.never_correct.length > 0) {
        neverList.innerHTML = insights.never_correct.slice(0, 5).map(item => `
            <div class="sample-item" onclick="openSampleInspector(${item.sample_id})">
                <span class="sample-id">#${item.sample_id}</span>
                <span class="sample-meta">True: ${item.true_label} → Pred: ${item.current_prediction}</span>
            </div>
        `).join('');
    } else {
        neverList.innerHTML = '<span class="empty-state">All samples correct at least once</span>';
    }
}

// ============================================================================
// Stability Chart
// ============================================================================

function initStabilityChart() {
    const ctx = document.getElementById('stabilityChart').getContext('2d');
    
    timelineState.charts.stability = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Stable Correct', 'Stable Wrong', 'Unstable', 'Late Learner'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: ['#238636', '#da3633', '#d29922', '#58a6ff'],
                borderColor: '#0d1117',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#8b949e', padding: 15 }
                }
            }
        }
    });
}

function updateStabilityChart() {
    const insights = timelineState.insights;
    if (!insights || !insights.stability_distribution) return;
    
    const dist = insights.stability_distribution;
    const chart = timelineState.charts.stability;
    
    chart.data.datasets[0].data = [
        dist.stable_correct || 0,
        dist.stable_wrong || 0,
        dist.unstable || 0,
        dist.late_learner || 0
    ];
    chart.update();
}

// ============================================================================
// Sample Inspector (Block B)
// ============================================================================

async function openSampleInspector(sampleId) {
    const modal = document.getElementById('sample-inspector-modal');
    modal.style.display = 'flex';
    
    // Fetch sample data
    const data = await fetchSampleTimeline(sampleId);
    if (!data) {
        alert('Could not load sample data');
        closeSampleInspector();
        return;
    }
    
    timelineState.currentSample = data;
    
    // Update header stats
    document.getElementById('inspector-sample-id').textContent = `#${data.sample_id}`;
    document.getElementById('inspector-true-label').textContent = data.true_label || '-';
    document.getElementById('inspector-flip-count').textContent = data.flip_count || 0;
    document.getElementById('inspector-stability').textContent = formatStability(data.stability_class);
    
    // Update charts
    updateInspectorCharts(data);
    
    // Update event history
    updateEventHistory(data.events || []);
}

function closeSampleInspector() {
    document.getElementById('sample-inspector-modal').style.display = 'none';
    timelineState.currentSample = null;
}

function formatStability(stability) {
    const map = {
        'stable_correct': '✓ Stable',
        'stable_wrong': '✗ Persistent Error',
        'unstable': '⚡ Unstable',
        'late_learner': '🕐 Late Learner',
        'unknown': '?'
    };
    return map[stability] || stability;
}

function updateInspectorCharts(data) {
    const events = data.events || [];
    if (!events.length) return;
    
    const epochs = events.map(e => e.epoch);
    const confidences = events.map(e => e.confidence);
    const predictions = events.map(e => e.predicted_label);
    const correctness = events.map(e => e.correct ? 1 : 0);
    
    // Prediction trajectory chart
    const predCtx = document.getElementById('predictionChart').getContext('2d');
    if (timelineState.charts.prediction) {
        timelineState.charts.prediction.destroy();
    }
    
    timelineState.charts.prediction = new Chart(predCtx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [{
                label: 'Correct',
                data: correctness,
                borderColor: '#238636',
                backgroundColor: 'rgba(35, 134, 54, 0.1)',
                fill: true,
                tension: 0.3,
                stepped: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    min: 0, 
                    max: 1,
                    ticks: { 
                        callback: v => v === 1 ? 'Correct' : 'Wrong',
                        color: '#8b949e'
                    },
                    grid: { color: '#30363d' }
                },
                x: { 
                    title: { display: true, text: 'Epoch', color: '#8b949e' },
                    grid: { color: '#30363d' }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
    
    // Confidence chart
    const confCtx = document.getElementById('confidenceChart').getContext('2d');
    if (timelineState.charts.confidence) {
        timelineState.charts.confidence.destroy();
    }
    
    timelineState.charts.confidence = new Chart(confCtx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [{
                label: 'Confidence',
                data: confidences,
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { 
                    min: 0, 
                    max: 1,
                    grid: { color: '#30363d' },
                    ticks: { color: '#8b949e' }
                },
                x: { 
                    title: { display: true, text: 'Epoch', color: '#8b949e' },
                    grid: { color: '#30363d' }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function updateEventHistory(events) {
    const container = document.getElementById('event-history-list');
    
    if (!events.length) {
        container.innerHTML = '<span class="empty-state">No events recorded</span>';
        return;
    }
    
    // Detect flips
    const flips = new Set();
    for (let i = 1; i < events.length; i++) {
        if (events[i].predicted_label !== events[i-1].predicted_label) {
            flips.add(i);
        }
    }
    
    container.innerHTML = events.map((e, idx) => `
        <div class="event-item ${e.correct ? 'correct' : 'wrong'}">
            <span class="epoch-badge">Epoch ${e.epoch}</span>
            <span>True: ${e.true_label}</span>
            <span>Pred: ${e.predicted_label}</span>
            <span>Conf: ${(e.confidence * 100).toFixed(1)}%</span>
            <span>${flips.has(idx) ? '<span class="flip-marker">⚡ FLIP</span>' : ''}</span>
        </div>
    `).join('');
}

// Close modal on backdrop click
document.getElementById('sample-inspector-modal')?.addEventListener('click', function(e) {
    if (e.target === this) {
        closeSampleInspector();
    }
});

// Keyboard handler
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeSampleInspector();
    }
});

// Initialize on load
document.addEventListener('DOMContentLoaded', initTimeline);
