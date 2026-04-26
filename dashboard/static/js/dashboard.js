// Dashboard JavaScript for YOLO Model Comparison

let allModels = [];
let filteredModels = [];
let selectedModels = new Set();
let selectedMetric = 'f1';
let chartType = 'bar';
let comparisonMode = 'individual';
let filters = {
    yoloVersions: new Set(),
    sizes: new Set(),
    architectures: new Set(),
    pixelFilters: new Set()
};
let mainChart = null;
let availableFilters = {};

// Metric definitions
const metricDefs = {
    'total_tp': { label: 'Total True Positives', isRate: false, color: '#4CAF50' },
    'total_fp': { label: 'Total False Positives', isRate: false, color: '#f44336' },
    'total_fn': { label: 'Total False Negatives', isRate: false, color: '#ff9800' },
    'precision': { label: 'Precision', isRate: true, color: '#2196F3' },
    'recall': { label: 'Recall', isRate: true, color: '#9C27B0' },
    'f1': { label: 'F1 Score', isRate: true, color: '#00BCD4' },
    'ap50': { label: 'AP@50', isRate: true, color: '#FF5722' }
};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    loadData();
});

// Load model data from API
async function loadData(refresh = false) {
    updateStatus('Loading...', true);
    
    try {
        const response = await fetch(`/api/models${refresh ? '?refresh=true' : ''}`);
        const data = await response.json();
        allModels = data.models;
        filteredModels = [...allModels];
        
        // Load filters
        const filtersResponse = await fetch('/api/filters');
        availableFilters = await filtersResponse.json();
        
        renderFilters();
        renderMetricSelector();
        renderModelList();
        updateChart();
        
        document.getElementById('last-updated').textContent = 
            `Last updated: ${new Date().toLocaleString()}`;
        updateStatus(`Loaded ${allModels.length} models`);
    } catch (error) {
        updateStatus('Error loading data: ' + error.message);
        console.error(error);
    }
}

// Refresh data on demand
function refreshData() {
    loadData(true);
}

// Render filter chips
function renderFilters() {
    // YOLO Versions
    const yoloContainer = document.getElementById('yolo-version-filters');
    yoloContainer.innerHTML = availableFilters.yolo_versions.map(v => 
        `<div class="filter-chip ${filters.yoloVersions.has(v) ? 'active' : ''}" 
             onclick="toggleFilter('yoloVersions', '${v}')">
            <input type="checkbox" ${filters.yoloVersions.has(v) ? 'checked' : ''}>
            YOLO ${v}
        </div>`
    ).join('');
    
    // Sizes
    const sizeContainer = document.getElementById('size-filters');
    sizeContainer.innerHTML = availableFilters.sizes.map(s => 
        `<div class="filter-chip ${filters.sizes.has(s) ? 'active' : ''}" 
             onclick="toggleFilter('sizes', '${s}')">
            <input type="checkbox" ${filters.sizes.has(s) ? 'checked' : ''}>
            ${s.toUpperCase()}
        </div>`
    ).join('');
    
    // Architectures
    const archContainer = document.getElementById('arch-filters');
    archContainer.innerHTML = availableFilters.architectures.map(a => 
        `<div class="filter-chip ${filters.architectures.has(a) ? 'active' : ''}" 
             onclick="toggleFilter('architectures', '${a}')">
            <input type="checkbox" ${filters.architectures.has(a) ? 'checked' : ''}>
            ${a.toUpperCase()}
        </div>`
    ).join('');
    
    // Pixel Filters
    const pixelContainer = document.getElementById('pixel-filters');
    pixelContainer.innerHTML = availableFilters.pixel_filters.map(p => 
        `<div class="filter-chip ${filters.pixelFilters.has(p) ? 'active' : ''}" 
             onclick="toggleFilter('pixelFilters', ${p})">
            <input type="checkbox" ${filters.pixelFilters.has(p) ? 'checked' : ''}>
            ${p === 0 ? 'All' : `Sub ${p}px`}
        </div>`
    ).join('');
}

// Toggle filter
function toggleFilter(filterType, value) {
    const set = filters[filterType];
    if (set.has(value)) {
        set.delete(value);
    } else {
        set.add(value);
    }
    applyFilters();
    renderFilters();
    renderModelList();
}

// Apply all filters
function applyFilters() {
    filteredModels = allModels.filter(m => {
        if (filters.yoloVersions.size > 0 && !filters.yoloVersions.has(m.yolo_version)) return false;
        if (filters.sizes.size > 0 && !filters.sizes.has(m.size)) return false;
        if (filters.architectures.size > 0 && !filters.architectures.has(m.architecture)) return false;
        if (filters.pixelFilters.size > 0 && !filters.pixelFilters.has(m.pixel_filter)) return false;
        return true;
    });
}

// Render metric selector
function renderMetricSelector() {
    const container = document.getElementById('metric-selector');
    container.innerHTML = availableFilters.metrics.map(m => {
        const def = metricDefs[m.key];
        return `
            <div class="metric-option ${selectedMetric === m.key ? 'active' : ''}"
                 onclick="selectMetric('${m.key}')">
                <span>${def.label}</span>
                <span style="font-size: 11px; color: #888;">${m.is_rate ? '0-1' : 'count'}</span>
            </div>
        `;
    }).join('');
}

// Select metric
function selectMetric(metric) {
    selectedMetric = metric;
    renderMetricSelector();
    updateChart();
}

// Render model list
function renderModelList() {
    const container = document.getElementById('model-list');
    container.innerHTML = filteredModels.map(m => `
        <div class="model-item ${selectedModels.has(m.id) ? 'selected' : ''}" 
             onclick="toggleModel('${m.id}')">
            <input type="checkbox" ${selectedModels.has(m.id) ? 'checked' : ''}>
            <div class="model-info">
                <div class="model-name">${m.label}</div>
                <div class="model-meta">YOLO ${m.yolo_version} · ${m.architecture} · ${m.pixel_filter === 0 ? 'All' : 'Sub ' + m.pixel_filter + 'px'}</div>
            </div>
        </div>
    `).join('');
}

// Toggle model selection
function toggleModel(modelId) {
    if (selectedModels.has(modelId)) {
        selectedModels.delete(modelId);
    } else {
        selectedModels.add(modelId);
    }
    renderModelList();
    updateChart();
    updateDetails();
}

// Select all visible models
function selectAll() {
    filteredModels.forEach(m => selectedModels.add(m.id));
    renderModelList();
    updateChart();
    updateDetails();
}

// Deselect all
function deselectAll() {
    selectedModels.clear();
    renderModelList();
    updateChart();
    updateDetails();
}

// Set comparison mode
function setComparisonMode(mode) {
    comparisonMode = mode;
    document.querySelectorAll('.comparison-mode').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.toLowerCase().includes(mode.replace('-', ' ')));
    });
    updateChart();
}

// Set chart type
function setChartType(type) {
    chartType = type;
    document.querySelectorAll('.chart-btn').forEach(btn => {
        if (btn.textContent.toLowerCase() === type) {
            btn.classList.add('active');
        } else if (['bar', 'line', 'radar', 'scatter'].includes(btn.textContent.toLowerCase())) {
            btn.classList.remove('active');
        }
    });
    updateChart();
}

// Group models for comparison modes
function getComparisonGroups() {
    const selected = allModels.filter(m => selectedModels.has(m.id));
    
    if (comparisonMode === 'individual') {
        return selected.map(m => ({
            label: m.label,
            models: [m],
            value: m.inference_metrics[selectedMetric] || 0
        }));
    }
    
    if (comparisonMode === 'p2p3') {
        // Group by base model name (without arch)
        const groups = {};
        selected.forEach(m => {
            const baseName = m.label.replace(/-p[23]-/, '-');
            if (!groups[baseName]) groups[baseName] = {};
            groups[baseName][m.architecture] = m;
        });
        
        return Object.entries(groups).map(([name, arches]) => ({
            label: name,
            p2: arches.p2?.inference_metrics[selectedMetric] || null,
            p3: arches.p3?.inference_metrics[selectedMetric] || null,
            models: [arches.p2, arches.p3].filter(Boolean)
        })).filter(g => g.p2 !== null || g.p3 !== null);
    }
    
    if (comparisonMode === 'pixels') {
        // Group by model and arch, compare pixel filters
        const groups = {};
        selected.forEach(m => {
            const key = `${m.yolo_version}-${m.size}-${m.architecture}`;
            if (!groups[key]) {
                groups[key] = {
                    label: `YOLO${m.yolo_version} ${m.size.toUpperCase()} ${m.architecture.toUpperCase()}`,
                    pixelValues: {}
                };
            }
            groups[key].pixelValues[m.pixel_filter] = m.inference_metrics[selectedMetric] || 0;
        });
        
        return Object.values(groups);
    }
    
    if (comparisonMode === 'yolo-versions') {
        // Group by size and arch, compare YOLO versions
        const groups = {};
        selected.forEach(m => {
            const key = `${m.size}-${m.architecture}-${m.pixel_filter}`;
            if (!groups[key]) {
                groups[key] = {
                    label: `${m.size.toUpperCase()} ${m.architecture.toUpperCase()} ${m.pixel_filter === 0 ? 'All' : 'Sub' + m.pixel_filter + 'px'}`,
                    versionValues: {}
                };
            }
            groups[key].versionValues[m.yolo_version] = m.inference_metrics[selectedMetric] || 0;
        });
        
        return Object.values(groups);
    }
    
    return [];
}

// Update main chart
function updateChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');
    
    if (mainChart) {
        mainChart.destroy();
    }
    
    if (selectedModels.size === 0) {
        document.getElementById('chart-title').textContent = 'Select models to compare';
        return;
    }
    
    const groups = getComparisonGroups();
    const metricDef = metricDefs[selectedMetric];
    document.getElementById('chart-title').textContent = `${metricDef.label} Comparison`;
    
    let chartConfig;
    
    if (chartType === 'bar') {
        chartConfig = createBarChart(groups, metricDef);
    } else if (chartType === 'line') {
        chartConfig = createLineChart(groups, metricDef);
    } else if (chartType === 'radar') {
        chartConfig = createRadarChart(groups);
    } else if (chartType === 'scatter') {
        chartConfig = createScatterChart(groups, metricDef);
    }
    
    mainChart = new Chart(ctx, chartConfig);
}

// Create bar chart configuration
function createBarChart(groups, metricDef) {
    const labels = groups.map(g => g.label);
    const data = groups.map(g => g.value !== undefined ? g.value : (g.p2 !== null ? g.p2 : g.p3));
    
    // For P2/P3 comparison mode, use grouped bars
    const datasets = [];
    if (comparisonMode === 'p2p3' && groups.some(g => g.p2 !== null || g.p3 !== null)) {
        datasets.push({
            label: 'P2',
            data: groups.map(g => g.p2),
            backgroundColor: 'rgba(102, 126, 234, 0.8)',
            borderColor: '#667eea',
            borderWidth: 2
        });
        datasets.push({
            label: 'P3',
            data: groups.map(g => g.p3),
            backgroundColor: 'rgba(118, 75, 162, 0.8)',
            borderColor: '#764ba2',
            borderWidth: 2
        });
    } else if (comparisonMode === 'pixels') {
        // Multiple datasets for different pixel filters
        const pixelFilters = [...new Set(groups.flatMap(g => Object.keys(g.pixelValues).map(Number)))].sort((a, b) => a - b);
        const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c'];
        
        pixelFilters.forEach((pf, i) => {
            datasets.push({
                label: pf === 0 ? 'All' : `Sub ${pf}px`,
                data: groups.map(g => g.pixelValues[pf] || null),
                backgroundColor: colors[i % colors.length],
                borderColor: colors[i % colors.length],
                borderWidth: 2
            });
        });
    } else if (comparisonMode === 'yolo-versions') {
        const versions = [...new Set(groups.flatMap(g => Object.keys(g.versionValues)))].sort();
        const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c'];
        
        versions.forEach((v, i) => {
            datasets.push({
                label: `YOLO ${v}`,
                data: groups.map(g => g.versionValues[v] || null),
                backgroundColor: colors[i % colors.length],
                borderColor: colors[i % colors.length],
                borderWidth: 2
            });
        });
    } else {
        datasets.push({
            label: metricDef.label,
            data: data,
            backgroundColor: metricDef.color,
            borderColor: metricDef.color,
            borderWidth: 2
        });
    }
    
    return {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: datasets.length > 1
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: metricDef.isRate ? 1.05 : undefined,
                    title: {
                        display: true,
                        text: metricDef.label
                    }
                }
            }
        }
    };
}

// Create line chart configuration
function createLineChart(groups, metricDef) {
    return {
        type: 'line',
        data: {
            labels: groups.map(g => g.label),
            datasets: [{
                label: metricDef.label,
                data: groups.map(g => g.value),
                borderColor: metricDef.color,
                backgroundColor: metricDef.color + '40',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: metricDef.isRate ? 1.05 : undefined,
                    title: {
                        display: true,
                        text: metricDef.label
                    }
                }
            }
        }
    };
}

// Create radar chart configuration
function createRadarChart(groups) {
    const metrics = ['precision', 'recall', 'f1', 'ap50'];
    const labels = metrics.map(m => metricDefs[m].label);
    
    const colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];
    
    const datasets = groups.slice(0, 6).map((g, i) => ({
        label: g.label,
        data: metrics.map(m => g.models[0]?.inference_metrics[m] || 0),
        borderColor: colors[i % colors.length],
        backgroundColor: colors[i % colors.length] + '30',
        borderWidth: 2
    }));
    
    return {
        type: 'radar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    };
}

// Create scatter chart configuration
function createScatterChart(groups, metricDef) {
    return {
        type: 'scatter',
        data: {
            datasets: groups.map((g, i) => ({
                label: g.label,
                data: [{
                    x: g.models[0]?.inference_metrics.precision || 0,
                    y: g.models[0]?.inference_metrics.recall || 0
                }],
                backgroundColor: `hsl(${(i * 60) % 360}, 70%, 50%)`
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Precision' },
                    min: 0,
                    max: 1
                },
                y: {
                    title: { display: true, text: 'Recall' },
                    min: 0,
                    max: 1
                }
            }
        }
    };
}

// Update details panel
function updateDetails() {
    const container = document.getElementById('details-content');
    const selected = allModels.filter(m => selectedModels.has(m.id));
    
    if (selected.length === 0) {
        container.innerHTML = `
            <div class="no-selection">
                <div class="no-selection-icon">📈</div>
                <p>Select a model to view detailed metrics</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '<div class="details-grid">' + selected.map(m => `
        <div class="details-section">
            <h4>${m.label}</h4>
            
            <table class="details-table">
                <tr><td>YOLO Version</td><td>${m.yolo_version}</td></tr>
                <tr><td>Architecture</td><td>${m.architecture}</td></tr>
                <tr><td>Pixel Filter</td><td>${m.pixel_filter === 0 ? 'All' : 'Sub ' + m.pixel_filter + 'px'}</td></tr>
            </table>
            
            <h4 style="margin-top: 15px;">Inference Metrics</h4>
            <table class="details-table">
                <tr><td>Precision</td><td>${(m.inference_metrics.precision || 0).toFixed(4)}</td></tr>
                <tr><td>Recall</td><td>${(m.inference_metrics.recall || 0).toFixed(4)}</td></tr>
                <tr><td>F1 Score</td><td>${(m.inference_metrics.f1 || 0).toFixed(4)}</td></tr>
                <tr><td>AP@50</td><td>${(m.inference_metrics.ap50 || 0).toFixed(4)}</td></tr>
                <tr><td>True Positives</td><td>${m.inference_metrics.total_tp || 0}</td></tr>
                <tr><td>False Positives</td><td>${m.inference_metrics.total_fp || 0}</td></tr>
                <tr><td>False Negatives</td><td>${m.inference_metrics.total_fn || 0}</td></tr>
            </table>
            
            ${m.training_metrics.best_map50 ? `
            <h4 style="margin-top: 15px;">Training Metrics</h4>
            <table class="details-table">
                <tr><td>Best mAP50</td><td>${m.training_metrics.best_map50.toFixed(4)}</td></tr>
                <tr><td>Best Epoch</td><td>${m.training_metrics.best_epoch}</td></tr>
                <tr><td>Final mAP50-95</td><td>${m.training_metrics.final_map50_95?.toFixed(4) || 'N/A'}</td></tr>
            </table>
            ` : ''}
            
            ${Object.keys(m.args).length > 0 ? `
            <h4 style="margin-top: 15px;">Training Args</h4>
            <table class="details-table">
                <tr><td>Epochs</td><td>${m.args.epochs || 'N/A'}</td></tr>
                <tr><td>Batch Size</td><td>${m.args.batch_size || 'N/A'}</td></tr>
                <tr><td>Learning Rate</td><td>${m.args.lr0 || 'N/A'}</td></tr>
                <tr><td>Optimizer</td><td>${m.args.optimizer || 'N/A'}</td></tr>
                <tr><td>Image Size</td><td>${m.args.img_size || 'N/A'}</td></tr>
                <tr><td>Patience</td><td>${m.args.patience || 'N/A'}</td></tr>
            </table>
            ` : ''}
        </div>
    `).join('') + '</div>';
}

// Export chart as image
function exportChart() {
    if (!mainChart) return;
    
    const link = document.createElement('a');
    link.download = `model-comparison-${selectedMetric}-${new Date().toISOString().slice(0,10)}.png`;
    link.href = mainChart.toBase64Image();
    link.click();
}

// Export data as CSV
function exportData() {
    const selected = allModels.filter(m => selectedModels.has(m.id));
    if (selected.length === 0) {
        alert('Please select at least one model to export');
        return;
    }
    
    const headers = ['Model', 'YOLO Version', 'Size', 'Architecture', 'Pixel Filter', 
                     'Precision', 'Recall', 'F1', 'AP50', 'TP', 'FP', 'FN',
                     'Best mAP50', 'Best Epoch'];
    
    const rows = selected.map(m => [
        m.label,
        m.yolo_version,
        m.size,
        m.architecture,
        m.pixel_filter,
        m.inference_metrics.precision,
        m.inference_metrics.recall,
        m.inference_metrics.f1,
        m.inference_metrics.ap50,
        m.inference_metrics.total_tp,
        m.inference_metrics.total_fp,
        m.inference_metrics.total_fn,
        m.training_metrics.best_map50,
        m.training_metrics.best_epoch
    ]);
    
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.download = `model-data-${new Date().toISOString().slice(0,10)}.csv`;
    link.href = URL.createObjectURL(blob);
    link.click();
}

// Update status bar
function updateStatus(text, loading = false) {
    const statusText = document.getElementById('status-text');
    statusText.innerHTML = loading ? `<span class="loading"></span> ${text}` : text;
}
