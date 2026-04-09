const API_URL = 'http://localhost:8000/api';

const app = {
    // Feature metadata cache
    metadata: null,
    
    // Default color mapping
    colors: {
        safe: '#3fb950',
        low: '#9d4edd',
        mod: '#d29922',
        high: '#f78166',
        crit: '#f85149'
    },

    // ──────────────────────────────────────────────────────────
    // INDEX PAGE METHODS
    // ──────────────────────────────────────────────────────────
    initIndexPage: async () => {
        const grid = document.getElementById('features-grid');
        const fillBtn = document.getElementById('fill-sample-btn');
        const form = document.getElementById('financial-form');
        
        if (!grid) return;
        
        try {
            // Fetch metadata exactly matching model features
            const response = await fetch(`${API_URL}/metadata`);
            const data = await response.json();
            app.metadata = data;
            
            // Build form
            grid.innerHTML = '';
            data.forEach((feature, index) => {
                const isRatio = feature.name.toLowerCase().includes('ratio') || feature.name.toLowerCase().includes('rate');
                const step = "any";
                
                grid.innerHTML += `
                    <div class="input-group">
                        <label title="${feature.name}">${feature.name}</label>
                        <input type="number" step="${step}" id="feat_${index}" required placeholder="Enter value">
                        <span class="validation-msg" id="msg_${index}"></span>
                    </div>
                `;
            });
            
            // Fill realistic data on button click
            fillBtn.addEventListener('click', () => {
                // Randomly choose to generate a "Good Company" or a "Bad Company"
                const generateHighRisk = Math.random() > 0.5;
                
                data.forEach((feature, index) => {
                    const el = document.getElementById(`feat_${index}`);
                    const fname = feature.name.toLowerCase();
                    
                    let val = 0;
                    
                    // Rules of thumb for good vs bad companies
                    if (fname.includes('debt') || fname.includes('liability') || fname.includes('borrow')) {
                        // High debt = bad
                        val = generateHighRisk ? 0.6 + Math.random() * 0.3 : 0.05 + Math.random() * 0.15;
                    } 
                    else if (fname.includes('profit') || fname.includes('cash') || fname.includes('income') || fname.includes('roa') || fname.includes('roe')) {
                        // High profit/cash = good
                        val = generateHighRisk ? 0.0 + Math.random() * 0.1 : 0.6 + Math.random() * 0.3;
                    }
                    else if (fname.includes('growth')) {
                        // High growth = good
                        val = generateHighRisk ? Math.random() * 0.1 : 0.4 + Math.random() * 0.5;
                    }
                    else {
                        // Random noise for neutral features
                        const span = feature.adjustment_range[1] - feature.adjustment_range[0];
                        val = feature.adjustment_range[0] + (Math.random() * span);
                        val = Math.abs(val);
                    }
                    
                    el.value = val.toFixed(4);
                });
                
                // Show a quick toast notification
                const type = generateHighRisk ? "Distressed/High-Risk" : "Healthy/Low-Risk";
                const btnOriginal = fillBtn.innerHTML;
                fillBtn.innerHTML = `<i class="fa-solid fa-check"></i> Filled ${type} Data`;
                setTimeout(() => { fillBtn.innerHTML = btnOriginal; }, 2000);
            });
            
            // Handle form submission
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                let isValid = true;
                const features = [];
                
                // Clear previous messages
                document.querySelectorAll('.validation-msg').forEach(el => el.textContent = '');
                
                for (let i = 0; i < data.length; i++) {
                    const el = document.getElementById(`feat_${i}`);
                    let val = parseFloat(el.value);
                    
                    if (isNaN(val)) {
                        document.getElementById(`msg_${i}`).textContent = `Please enter a valid number`;
                        isValid = false;
                    }
                    features.push(val);
                }
                
                if (!isValid) {
                    alert('Please fix the validation errors before analyzing.');
                    return;
                }
                
                // Save to local storage and redirect
                localStorage.setItem('finrisk_features', JSON.stringify(features));
                window.location.href = 'dashboard.html';
            });
            
        } catch (error) {
            grid.innerHTML = `<div class="error-msg">Failed to load features: ${error.message}</div>`;
        }
    },

    // ──────────────────────────────────────────────────────────
    // DASHBOARD PAGE METHODS
    // ──────────────────────────────────────────────────────────
    initDashboardPage: async () => {
        const dashboard = document.getElementById('dashboard-container');
        if (!dashboard) return;
        
        const rawFeatures = localStorage.getItem('finrisk_features');
        if (!rawFeatures) {
            window.location.href = 'index.html';
            return;
        }
        
        const features = JSON.parse(rawFeatures);
        document.getElementById('loading-overlay').style.display = 'flex';
        
        try {
            // Run Prediction and SHAP in parallel
            const [predRes, shapRes] = await Promise.all([
                fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({features})
                }),
                fetch(`${API_URL}/explain`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({features})
                })
            ]);
            
            const predData = await predRes.json();
            const shapData = await shapRes.json();
            
            // Setup Dashboard View
            document.getElementById('loading-overlay').style.display = 'none';
            dashboard.style.display = 'grid'; // .layout-dashboard uses grid
            
            app.renderRiskSummary(predData);
            app.renderShapChart(shapData, predData.risk_score);
            
            // Setup RL Button
            document.getElementById('run-rl-btn').addEventListener('click', () => {
                app.runRlSimulation(features);
            });
            
            // Setup Advisor Button
            document.getElementById('run-advisor-btn').addEventListener('click', () => {
                app.runAiAdvisor(features);
            });
            
        } catch (error) {
            alert('Failed to analyze company risk: ' + error.message);
        }
    },
    
    getRiskColor: (category) => {
        if(category.includes('Safe')) return app.colors.safe;
        if(category.includes('Low'))  return app.colors.low;
        if(category.includes('Moderate')) return app.colors.mod;
        if(category.includes('High')) return app.colors.high;
        return app.colors.crit;
    },

    renderRiskSummary: (data) => {
        // Enforce 0-100 logic explicitly
        const boundedProbability = Math.max(0, Math.min(1, data.bankruptcy_probability));
        const riskScoreNum = boundedProbability * 100;
        const score = riskScoreNum.toFixed(1);
        
        // Use exact logic: 0-20, 20-40, 40-60, 60-80, 80-100
        let category = 'Safe';
        let color = app.colors.safe;
        
        if (riskScoreNum >= 80) { category = 'Critical Risk'; color = app.colors.crit; }
        else if (riskScoreNum >= 60) { category = 'High Risk'; color = app.colors.high; }
        else if (riskScoreNum >= 40) { category = 'Moderate Risk'; color = app.colors.mod; }
        else if (riskScoreNum >= 20) { category = 'Low Risk'; color = app.colors.low; }
        
        document.getElementById('risk-score-val').textContent = score;
        document.getElementById('risk-score-val').style.color = color;
        
        const circle = document.getElementById('risk-circle');
        circle.style.borderColor = color;
        circle.style.boxShadow = `0 0 20px ${color}40`;
        
        document.getElementById('risk-category-name').textContent = category;
        document.getElementById('risk-category-name').style.color = color;
        
        document.getElementById('risk-probability-val').textContent = score + '%';
        
        const bar = document.getElementById('risk-bar-fill');
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = `${score}%`;
            bar.style.backgroundColor = color;
        }, 300);
    },

    renderShapChart: (data, riskScore) => {
        const ctx = document.getElementById('shap-chart').getContext('2d');
        
        // Take top 5 risk factors and top 5 protective factors
        const topRisk = data.top_risk_factors.slice(0, 5);
        const topProtective = data.top_protective_factors.slice(0, 5);
        
        // Combine them so risk factors are at the top, protective at bottom
        const combined = [...topRisk, ...topProtective];
        
        const labels = combined.map(c => 
            c.feature.length > 35 ? c.feature.substring(0, 32) + '...' : c.feature
        );
        const values = combined.map(c => c.value);
        const bgColors = values.map(v => v > 0 ? '#f85149' : '#9d4edd');
        
        if (window.shapChartInstance) window.shapChartInstance.destroy();
        
        window.shapChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Impact on Risk Score',
                    data: values,
                    backgroundColor: bgColors,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const val = ctx.raw;
                                return val > 0 ? `+${val.toFixed(4)} (Increases Risk)` : `${val.toFixed(4)} (Decreases Risk)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#8b949e' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: '#c9d1d9',
                            font: { size: 10 }
                        }
                    }
                }
            }
        });
        
        // Generate SHAP Summary Text
        const summaryText = document.getElementById('shap-summary-text');
        
        let text = "";
        
        if (riskScore > 40) {
            text = `This company faces elevated bankruptcy probability primarily driven by its <strong>${topRisk[0].feature.toLowerCase()}</strong> and <strong>${topRisk[1].feature.toLowerCase()}</strong>, which are heavily penalizing its standing. `;
            if (topProtective.length > 0) {
                text += `These risks are only partially mitigated by a strong <strong>${topProtective[0].feature.toLowerCase()}</strong> metric, which acts as the company's main financial anchor.`;
            } else {
                text += `The company lacks any significant protective financial buffers to offset these vulnerabilities.`;
            }
        } else {
            text = `This company maintains a healthy financial profile securely anchored by a strong <strong>${topProtective[0].feature.toLowerCase()}</strong> and favorable <strong>${topProtective[1].feature.toLowerCase()}</strong>. `;
            if (topRisk.length > 0) {
                text += `While it remains broadly safe, the ML model flags its <strong>${topRisk[0].feature.toLowerCase()}</strong> as the primary area of friction that management should monitor.`;
            }
        }
        
        summaryText.innerHTML = text;
    },

    runRlSimulation: async (features) => {
        const btn = document.getElementById('run-rl-btn');
        const loading = document.getElementById('rl-loading');
        const results = document.getElementById('rl-results');
        
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Running...';
        results.style.display = 'none';
        loading.style.display = 'flex';
        
        try {
            const response = await fetch(`${API_URL}/strategy`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({features})
            });
            const data = await response.json();
            
            loading.style.display = 'none';
            results.style.display = 'block';
            
            if (data.skip_rl) {
                results.innerHTML = `
                    <div style="text-align:center; padding: 2rem; color: var(--risk-safe);">
                        <i class="fa-solid fa-shield-check" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                        <h3>${data.message}</h3>
                    </div>
                `;
                btn.innerHTML = '<i class="fa-solid fa-check"></i> Analysis Complete';
                return;
            }
            
            // Re-show stats block just in case it was overwritten by skip_rl previously
            if(results.innerHTML.includes('fa-shield-check')) {
                // If the user refreshed or changed inputs without reloading the page, we'd need a clean DOM.
                // Assuming standard reload for now, but just show block.
            }
            
            // Populate stats
            document.getElementById('rl-initial-risk').textContent = data.initial_risk.toFixed(1) + '%';
            document.getElementById('rl-final-risk').textContent = data.final_risk.toFixed(1) + '%';
            
            const reduction = data.initial_risk - data.final_risk;
            const reductionLabel = document.getElementById('rl-reduction-label');
            const reductionValue = document.getElementById('rl-reduction');
            
            if (reduction >= 0) {
                const pct = (reduction / Math.max(0.0001, data.initial_risk)) * 100;
                reductionLabel.textContent = "Risk Reduction";
                reductionValue.className = "stat-value text-green";
                reductionValue.textContent = pct.toFixed(1) + '%';
            } else {
                const pct = (-reduction / Math.max(0.0001, data.initial_risk)) * 100;
                reductionLabel.textContent = "Risk Increase";
                reductionValue.className = "stat-value text-red";
                reductionValue.textContent = '+' + pct.toFixed(1) + '%';
            }
            // Generate Strategy Paragraph
            const paragraphContainer = document.getElementById('rl-strategy-paragraph');
            
            const actionCounts = {};
            let uniqueActionsInOrder = [];
            
            data.history.forEach((h, i) => {
                if (i === 0) return; // skip initial state
                
                const baseAction = h.action_str;
                actionCounts[baseAction] = (actionCounts[baseAction] || 0) + 1;
                if (!uniqueActionsInOrder.includes(baseAction)) {
                    uniqueActionsInOrder.push(baseAction);
                }
            });
            
            // Generate a natural language paragraph
            let paragraphHTML = "";
            
            if (uniqueActionsInOrder.length > 0) {
                paragraphHTML += `To rapidly de-risk the company's financial profile over the next 10 quarters, the AI recommends immediately securing <strong>${uniqueActionsInOrder[0].toLowerCase()}</strong>`;
                
                if (actionCounts[uniqueActionsInOrder[0]] > 2) {
                    paragraphHTML += ` as a long-term, sustained focus. `;
                } else {
                    paragraphHTML += ` as an initial corrective measure. `;
                }
                
                if (uniqueActionsInOrder.length > 1) {
                    paragraphHTML += `Subsequently, the company must also prioritize <strong>${uniqueActionsInOrder[1].toLowerCase()}</strong>. `;
                }
                
                if (uniqueActionsInOrder.length > 2) {
                    const remaining = uniqueActionsInOrder.slice(2).map(a => `<strong>${a.toLowerCase()}</strong>`).join(', and ');
                    paragraphHTML += `Further optimizations should include ${remaining}. `;
                }
                
                paragraphHTML += `By strictly adhering to this sequential strategy across the simulated timeline, the company is projected to drop its bankruptcy probability from ${data.initial_risk.toFixed(1)}% down to ${data.final_risk.toFixed(1)}%.`;
            } else {
                paragraphHTML = "No actionable steps were required.";
            }
            
            paragraphContainer.innerHTML = paragraphHTML;
            
            // Generate Summary text
            const sortedActions = Object.entries(actionCounts).sort((a,b) => b[1] - a[1]);
            const summaryContainer = document.getElementById('rl-summary-container');
            const summaryText = document.getElementById('rl-summary-text');
            
            if (sortedActions.length > 0) {
                const top1 = sortedActions[0][0].toLowerCase();
                const top2 = sortedActions.length > 1 ? sortedActions[1][0].toLowerCase() : "";
                
                let text = `This strategy focuses primarily on <strong>${top1}</strong>`;
                if (top2) text += ` and correspondingly aims to <strong>${top2}</strong>`;
                text += `, which are identified as the most highly leveraged pathways to minimize bankruptcy probability over the simulated 10-quarter horizon.`;
                
                summaryText.innerHTML = text;
                summaryContainer.style.display = 'block';
            }
            
            // Draw RL chart
            const ctx = document.getElementById('rl-chart').getContext('2d');
            const labels = data.history.map(h => `Q${h.step}`);
            const probs = data.history.map(h => h.probability);
            
            const isImproving = data.final_risk < data.initial_risk;
            const lineColor = isImproving ? '#3fb950' : '#f85149';
            const bgColor = isImproving ? 'rgba(63, 185, 80, 0.15)' : 'rgba(248, 81, 73, 0.15)';
            
            // Destroy existing chart if RL btn clicked twice
            if(window.rlChartInstance) window.rlChartInstance.destroy();
            
            window.rlChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Projected Risk',
                        data: probs,
                        borderColor: lineColor,
                        backgroundColor: bgColor,
                        pointBackgroundColor: '#fff',
                        pointBorderColor: lineColor,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        fill: true,
                        tension: 0.4 // Smoothed curve
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            min: 0,
                            suggestedMax: Math.max(5, Math.max(...probs) * 1.2),
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#8b949e', callback: v => v.toFixed(0) + '%' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#8b949e' }
                        }
                    }
                }
            });
            
            btn.innerHTML = '<i class="fa-solid fa-check"></i> Optimization Complete';
            
        } catch (error) {
            loading.style.display = 'none';
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Simulation';
            alert('Failed to run RL simulation: ' + error.message);
        }
    },
    
    // ──────────────────────────────────────────────────────────
    // AI ADVISOR METHODS
    // ──────────────────────────────────────────────────────────
    runAiAdvisor: async (features) => {
        const btn = document.getElementById('run-advisor-btn');
        const loading = document.getElementById('advisor-loading');
        const results = document.getElementById('advisor-results');
        
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> AI Agent Working...';
        results.style.display = 'none';
        loading.style.display = 'block';
        
        try {
            const response = await fetch(`${API_URL}/advisor`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({features})
            });
            const data = await response.json();
            
            loading.style.display = 'none';
            results.style.display = 'block';
            
            if (data.error) {
                throw new Error(data.error);
            }
            if (data.detail) {
                // Catches FastAPI HTTPExceptions (like Rate Limits and server crashes)
                throw new Error(data.detail);
            }

            const ra = data.company_risk_assessment;
            const riskScore = ra.risk_score ?? 0;
            const riskCategory = ra.risk_category ?? 'Unknown';
            const bankProb = ra.bankruptcy_probability ?? 0;
            const projRisk = data.projected_risk_after_strategy;

            // ── Banner ──────────────────────────────────────────────
            const banner = document.getElementById('advisor-banner');
            const bannerTitle = document.getElementById('advisor-banner-title');
            const bannerSubtitle = document.getElementById('advisor-banner-subtitle');
            const bannerIcon = document.getElementById('advisor-banner-icon');
            
            const isHighRisk = riskScore >= 50;
            const isMedRisk = riskScore >= 20 && riskScore < 50;
            const bannerColor = isHighRisk ? '#f85149' : isMedRisk ? '#ffc400' : '#3fb950';
            const bannerBg = isHighRisk ? 'rgba(248,81,73,0.08)' : isMedRisk ? 'rgba(255,196,0,0.08)' : 'rgba(63,185,80,0.08)';
            const bannerBorder = isHighRisk ? 'rgba(248,81,73,0.2)' : isMedRisk ? 'rgba(255,196,0,0.2)' : 'rgba(63,185,80,0.2)';
            
            banner.style.background = bannerBg;
            banner.style.borderBottom = `1px solid ${bannerBorder}`;
            bannerIcon.textContent = isHighRisk ? '🚨' : isMedRisk ? '⚠️' : '✅';
            bannerTitle.textContent = isHighRisk ? 'High Bankruptcy Risk Detected — Immediate Action Required' 
                                    : isMedRisk ? 'Moderate Risk Identified — Strategic Intervention Recommended'
                                    : 'Company Appears Financially Healthy';
            bannerTitle.style.color = bannerColor;
            bannerSubtitle.textContent = `Agent analyzed ${features.length} financial indicators and orchestrated 3 specialized tools to generate this report.`;

            // ── 4 Metric Cards ──────────────────────────────────────
            const categoryEl = document.getElementById('advisor-risk-category');
            categoryEl.textContent = riskCategory;
            categoryEl.style.color = bannerColor;

            const scoreEl = document.getElementById('advisor-risk-score');
            scoreEl.textContent = riskScore.toFixed(1);
            scoreEl.style.color = bannerColor;

            const probEl = document.getElementById('advisor-bankruptcy-prob');
            probEl.textContent = (bankProb * 100).toFixed(1) + '%';
            probEl.style.color = bannerColor;

            const projEl = document.getElementById('advisor-projected-risk');
            if (projRisk !== null && projRisk !== undefined) {
                projEl.textContent = projRisk.toFixed(1);
            } else {
                projEl.textContent = 'N/A';
                projEl.style.color = '#8b949e';
            }

            // ── Risk Drivers (as chips) ────────────────────────────
            const driversList = document.getElementById('advisor-drivers-list');
            driversList.innerHTML = '';
            const drivers = data.risk_drivers || [];
            if (drivers.length > 0) {
                drivers.forEach((driver, idx) => {
                    const chip = document.createElement('div');
                    chip.style.cssText = `display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.6rem 0.8rem; background: rgba(248,81,73,0.08); border: 1px solid rgba(248,81,73,0.2); border-radius: 6px; font-size: 0.88rem; color: #c9d1d9; line-height: 1.4;`;
                    chip.innerHTML = `<span style="color:#f85149; font-weight:700; flex-shrink:0;">⚠️</span><span>${driver.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</span>`;
                    driversList.appendChild(chip);
                });
            } else {
                driversList.innerHTML = '<div style="color:#8b949e; font-size:0.88rem; font-style:italic;">No significant high-risk drivers identified.</div>';
            }

            // ── Strategy Steps (numbered timeline) ─────────────────
            const strategyList = document.getElementById('advisor-strategy-list');
            strategyList.innerHTML = '';
            const steps = data.recommended_strategy || [];
            if (steps.length > 0) {
                steps.forEach((step, idx) => {
                    const stepDiv = document.createElement('div');
                    stepDiv.style.cssText = `display: flex; align-items: flex-start; gap: 0.8rem;`;
                    stepDiv.innerHTML = `
                        <div style="min-width: 24px; height: 24px; border-radius: 50%; background: rgba(63,185,80,0.15); border: 1px solid rgba(63,185,80,0.4); color: #3fb950; font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px;">${idx + 1}</div>
                        <div style="font-size: 0.88rem; color: #c9d1d9; line-height: 1.5; padding-top: 0.1rem;">${step.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#eee">$1</strong>')}</div>
                    `;
                    strategyList.appendChild(stepDiv);
                });
            } else {
                strategyList.innerHTML = '<div style="color:#8b949e; font-size:0.88rem; font-style:italic;">Company is financially healthy — no corrective actions required.</div>';
            }

            // ── Risk Reduction Bar ──────────────────────────────────
            if (projRisk !== null && projRisk !== undefined && riskScore > 0) {
                const reductionRow = document.getElementById('advisor-risk-reduction-row');
                if (reductionRow) {
                    reductionRow.style.display = 'block';
                    const reduction = Math.max(0, riskScore - projRisk);
                    const reductionPct = Math.min(100, (reduction / riskScore) * 100);
                    const pctEl = document.getElementById('advisor-reduction-pct');
                    if (pctEl) pctEl.textContent = reductionPct.toFixed(1) + '% reduction projected';
                    setTimeout(() => {
                        const barEl = document.getElementById('advisor-reduction-bar');
                        if (barEl) barEl.style.width = reductionPct + '%';
                    }, 100);
                }
            }

            // ── Agent Reasoning Log ─────────────────────────────────
            const logEl = document.getElementById('advisor-log');
            const timestamp = new Date().toLocaleTimeString();
            logEl.textContent = [
                `[${timestamp}] ▶ Agent initialized with LLaMA3-70B (Groq)`,
                `[${timestamp}] ▶ Tool 1: predict_bankruptcy_risk → risk_score=${riskScore.toFixed(1)}, category="${riskCategory}"`,
                riskScore >= 20 ? `[${timestamp}] ▶ Tool 2: generate_shap_explanation → ${drivers.length} drivers identified` : `[${timestamp}] ✓ Risk < 20: SHAP explanation skipped`,
                riskScore >= 40 ? `[${timestamp}] ▶ Tool 3: run_rl_strategy → projected_risk=${projRisk !== null ? projRisk.toFixed(1) : 'N/A'}` : `[${timestamp}] ✓ Risk < 40: RL strategy skipped`,
                `[${timestamp}] ▶ Generating final advisory report...`,
                `[${timestamp}] ✅ Report complete — ${steps.length} strategy steps, ${drivers.length} risk drivers`
            ].join('\n');

            btn.innerHTML = '<i class="fa-solid fa-check"></i> Report Generated';
            btn.style.color = '#3fb950';
            btn.style.borderColor = '#3fb950';
            
        } catch (error) {
            loading.style.display = 'none';
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i>&nbsp; Generate AI Advisory Report';
            alert('Failed to generate AI Advisory Report: ' + error.message);
        }
    }
};
