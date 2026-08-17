import React, { useState, useRef } from 'react';
import axios from 'axios';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';
import {
  HeartPulse, Activity, Users, BookOpen, Database, HeartCrack, BarChart3,
  Settings as SettingsIcon, Search, Bell, FileUp, Download, CheckCircle, Upload
} from 'lucide-react';

import ECGChart from './components/ECGChart';
import PatientRecords from './components/PatientRecords';
import ClinicalGuidelines from './components/ClinicalGuidelines';
import VectorDBStats from './components/VectorDBStats';
import Settings from './components/Settings';

function App() {
  const [currentView, setCurrentView] = useState('dashboard');

  // Dashboard state
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // PTB-XL source-dataset stats
  const [ptbxlStats, setPtbxlStats] = useState(null);
  const [ptbxlLoading, setPtbxlLoading] = useState(false);

  const loadPtbxlStats = async () => {
    setPtbxlLoading(true);
    try {
      const res = await axios.get('http://localhost:8000/api/ecg/ptbxl-eda');
      setPtbxlStats(res.data);
    } catch (err) {
      console.error('Failed to load PTB-XL stats', err);
    } finally {
      setPtbxlLoading(false);
    }
  };

  // Reference for PDF export area
  const reportRef = useRef(null);

  const analyzeECG = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:8000/api/analyze', {});
      setResults(response.data);
    } catch (err) {
      setError("Failed to connect to DeepCardio-RAG backend. Is it running on port 8000?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const exportPDF = async () => {
    if (!reportRef.current || !results) return;

    try {
      const canvas = await html2canvas(reportRef.current, { scale: 2, useCORS: true });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.setFontSize(16);
      pdf.text("DeepCardio-RAG Clinical Diagnostic Report", 15, 15);
      pdf.addImage(imgData, 'PNG', 0, 25, pdfWidth, pdfHeight);

      pdf.setFontSize(10);
      pdf.setTextColor(150);
      pdf.text(`Generated exactly: ${new Date().toLocaleString()}`, 15, pdf.internal.pageSize.getHeight() - 10);

      pdf.save(`DeepCardio-RAG_Report_${new Date().getTime()}.pdf`);
    } catch (err) {
      console.error("PDF Export failed", err);
      alert("Failed to export PDF.");
    }
  };

  const renderView = () => {
    switch (currentView) {
      case 'patients': return <PatientRecords />;
      case 'guidelines': return <ClinicalGuidelines />;
      case 'vectordb': return <VectorDBStats />;
      case 'settings': return <Settings />;
      default: return renderDashboard();
    }
  };

  const renderDashboard = () => (
    <>
      <header className="top-header">
        <div className="search-bar">
          <Search size={18} />
          <input type="text" placeholder="Search patient ID, name, or condition..." />
        </div>
        <div className="header-actions">
          <button className="icon-btn"><Bell size={18} /></button>
          <button className="primary-btn" onClick={analyzeECG} disabled={loading}>
            <FileUp size={18} />
            {loading ? "Processing via AI..." : "Upload & Analyze ECG"}
          </button>
        </div>
      </header>

      <div className="dashboard-grid">
        <div className="col-main">
          {/* Visualizer Component */}
          <div className="card">
            <div className="card-header">
              <h3><Activity size={20} color="#0f62fe" /> Real-time 12-Lead ECG Analysis</h3>
              <div className="status-badge pulse-anim">Monitoring</div>
            </div>
            <div className="chart-container">
              <ECGChart isAnalyzing={loading} />
            </div>
            <div className="action-bar">
              <div className="ecg-stats">
                <span className="stat-pill"><strong>HR:</strong> 72 bpm</span>
                <span className="stat-pill"><strong>PR:</strong> 160 ms</span>
                <span className="stat-pill"><strong>QRS:</strong> 90 ms</span>
                <span className="stat-pill"><strong>QTc:</strong> 410 ms</span>
              </div>
            </div>
          </div>

          {/* Database Retrieval Component */}
          <div className="card rag-context-card">
            <div className="card-header">
              <h3><BookOpen size={20} color="#0f62fe" /> Retrieved Clinical Knowledge (RAG)</h3>
              <span className="db-badge">Milvus (Demo Guidelines)</span>
            </div>
            <div className="rag-content">
              {!results && !loading && (
                <div className="empty-state">
                  <Database size={40} />
                  <p>Run analysis to retrieve relevant guidelines from vector database.</p>
                </div>
              )}
              {loading && <p className="loading-text" style={{ padding: '1rem', color: 'var(--primary)' }}>Vector Search in progress...</p>}
              {results && results.retrieved_guidelines.map((gdl, i) => (
                <div key={i} className="rag-item">
                  <strong>Source Guideline/Case:</strong><br />
                  {gdl}
                </div>
              ))}
            </div>
          </div>

          {/* PTB-XL Source Dataset Card */}
          <div className="card">
            <div className="card-header">
              <h3><Database size={20} color="#0f62fe" /> Source Dataset — PTB-XL (PhysioNet)</h3>
              <button className="outline-btn" onClick={loadPtbxlStats} disabled={ptbxlLoading}>
                <BarChart3 size={16} /> {ptbxlLoading ? 'Loading...' : 'Load Stats'}
              </button>
            </div>
            <p style={{ padding: '0.5rem 1.5rem 0', color: 'var(--text-light)', fontSize: '0.85rem' }}>
              "Upload &amp; Analyze ECG" samples a <strong>real clinical 12-lead ECG</strong> from PTB-XL —
              not synthetic noise — and shows the ground-truth diagnosis alongside the AI-generated report.
            </p>
            {ptbxlStats && (
              <div className="stats-row" style={{ padding: '1rem 1.5rem' }}>
                <span className="stat-pill"><strong>Records:</strong> {(ptbxlStats.total_records || 0).toLocaleString()}</span>
                <span className="stat-pill"><strong>NORM:</strong> {ptbxlStats.superclass_distribution?.NORM?.count?.toLocaleString() ?? '—'}</span>
                <span className="stat-pill"><strong>MI:</strong> {ptbxlStats.superclass_distribution?.MI?.count?.toLocaleString() ?? '—'}</span>
                <span className="stat-pill"><strong>STTC:</strong> {ptbxlStats.superclass_distribution?.STTC?.count?.toLocaleString() ?? '—'}</span>
                <span className="stat-pill">
                  <strong>CD+HYP:</strong> {((ptbxlStats.superclass_distribution?.CD?.count || 0) + (ptbxlStats.superclass_distribution?.HYP?.count || 0)).toLocaleString()}
                </span>
                <span className="stat-pill">
                  <strong>Real signals:</strong> {ptbxlStats.real_signals_available ? 'Yes' : 'No (demo stats)'}
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="col-side">
          <div className="card ai-report-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div className="card-header">
              <h3><Upload size={20} color="#0f62fe" /> Diagnostic Report</h3>
              <div className="engine-badge">Transformer-Generated</div>
            </div>

            {loading && (
              <div className="loading-overlay" style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <div className="spinner"></div>
                <p>DeepCardio-RAG is reasoning...</p>
              </div>
            )}

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            {/* AREA EXPORTED TO PDF */}
            <div ref={reportRef} className="pdf-export-area" style={{ flex: 1, display: loading ? 'none' : 'block' }}>
              <div className="report-content" style={{ minHeight: '150px' }}>
                {!results && !loading && !error && (
                  <div className="empty-state">
                    <p>Report will appear here after analysis.</p>
                  </div>
                )}
                {results && (
                  <div>
                    {results.source && (
                      <div style={{
                        display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap',
                        padding: '0 0 14px', marginBottom: '12px', borderBottom: '1px solid var(--border)',
                        fontSize: '0.85rem'
                      }}>
                        <span className={`risk-badge ${results.source.real_signal ? 'success' : 'risk-low'}`} style={{ margin: 0 }}>
                          {results.source.real_signal ? <HeartCrack size={16} /> : <Database size={16} />}
                          <span>
                            {results.source.real_signal ? 'Real PTB-XL Record' : 'Synthetic Fallback'}
                            {results.source.ecg_id != null ? ` · ECG #${results.source.ecg_id}` : ''}
                          </span>
                        </span>
                        <span style={{ color: 'var(--text-light)' }}>
                          <strong>Ground truth:</strong> {results.source.ground_truth_label || results.source.ground_truth_superclass || '—'}
                          {(results.source.all_superclasses || []).length > 1
                            ? ` (${results.source.all_superclasses.join(', ')})`
                            : ''}
                        </span>
                      </div>
                    )}
                    <h4 style={{ marginBottom: "10px", color: "#0f62fe", fontWeight: 600 }}>Generated Hypothesis:</h4>
                    <div style={{ lineHeight: 1.6, whiteSpace: 'pre-wrap', fontFamily: 'Inter, system-ui', color: 'var(--text-main)' }}>
                      {results.report}
                    </div>
                  </div>
                )}
              </div>

              {results && (
                <div className="report-metrics">
                  <div className="metric">
                    <span className="label">Inference Time:</span>
                    <span className="value success">{results.inference_time_seconds}s</span>
                  </div>
                  <div className="metric">
                    <span className="label">Encoder:</span>
                    <span className="value">ECGTransformerMoE (prototype)</span>
                  </div>
                  <div className="metric">
                    <span className="label">Model Weights:</span>
                    <span className="value" style={{color:'#e07000'}}>Random (not trained)</span>
                  </div>
                  <div className="metric">
                    <span className="label">Report Quality:</span>
                    <span className="value" style={{color:'#e07000'}}>Demo only — not validated</span>
                  </div>
                </div>
              )}
            </div>

            {results && !loading && (
              <div className="report-actions">
                <button className="outline-btn" onClick={exportPDF}>
                  <Download size={18} /> Export PDF
                </button>
                <button className="primary-btn">
                  <CheckCircle size={18} /> Sign & Save
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );

  return (
    <div className="app-container">
      {/* Sidebar Layout */}
      <aside className="sidebar">
        <div className="logo-area">
          <HeartPulse className="logo-icon" size={28} />
          <h2>DeepCardio<span>RAG</span></h2>
        </div>

        <nav className="nav-menu">
          <a href="#" className={`nav-item ${currentView === 'dashboard' ? 'active' : ''}`} onClick={() => setCurrentView('dashboard')}>
            <Activity size={20} /> Dashboard
          </a>
          <a href="#" className={`nav-item ${currentView === 'patients' ? 'active' : ''}`} onClick={() => setCurrentView('patients')}>
            <Users size={20} /> Patient Records
          </a>
          <a href="#" className={`nav-item ${currentView === 'guidelines' ? 'active' : ''}`} onClick={() => setCurrentView('guidelines')}>
            <BookOpen size={20} /> Clinical Guidelines
          </a>
          <a href="#" className={`nav-item ${currentView === 'vectordb' ? 'active' : ''}`} onClick={() => setCurrentView('vectordb')}>
            <Database size={20} /> Vector DB Stats
          </a>
          <a href="#" className={`nav-item ${currentView === 'settings' ? 'active' : ''}`} onClick={() => setCurrentView('settings')}>
            <SettingsIcon size={20} /> Settings
          </a>
        </nav>

        <div className="sidebar-footer">
          <div className="doctor-profile">
            <div className="avatar">MD</div>
            <div className="info">
              <strong>Physician Name</strong>
              <span>Configure in Settings</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Layout Area */}
      <main className="main-content">
        {renderView()}
      </main>
    </div>
  );
}

export default App;
