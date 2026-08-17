import React from 'react';
import { Settings as SettingsIcon, Shield, Bell, Cpu, Link } from 'lucide-react';

const Settings = () => {
    return (
        <div className="view-container" style={{ padding: '2rem', height: '100%', overflowY: 'auto' }}>
            <div className="view-header" style={{ marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <SettingsIcon color="var(--primary)" />
                    System Configuration
                </h2>
                <p style={{ color: 'var(--text-muted)' }}>Manage your DeepCardio-RAG instance, API endpoints, and layout preferences.</p>
            </div>

            <div className="dashboard-grid" style={{ padding: 0, gap: '1.5rem', gridTemplateColumns: 'minmax(300px, 1fr) 2fr' }}>

                {/* Settings Navigation Menu */}
                <div className="card" style={{ alignSelf: 'start', padding: '1rem' }}>
                    <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <a href="#" className="nav-item active" style={{ background: '#f0f4ff', color: 'var(--primary)' }}>
                            <Cpu size={18} /> Model Configuration
                        </a>
                        <a href="#" className="nav-item" style={{ color: 'var(--text-main)' }}>
                            <Link size={18} /> API Endpoints
                        </a>
                        <a href="#" className="nav-item" style={{ color: 'var(--text-main)' }}>
                            <Shield size={18} /> Privacy & Compliance
                        </a>
                        <a href="#" className="nav-item" style={{ color: 'var(--text-main)' }}>
                            <Bell size={18} /> Notifications
                        </a>
                    </nav>
                </div>

                {/* Settings Content Area */}
                <div className="card" style={{ padding: '2rem' }}>
                    <h3 style={{ fontSize: '1.2rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '1rem' }}>
                        Model Configuration
                    </h3>

                    <div style={{ marginBottom: '2rem' }}>
                        <label style={{ display: 'block', fontWeight: 600, marginBottom: '8px' }}>Large Language Model Architecture</label>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '12px' }}>
                            Select the generative transformer model used for reasoning over the ECG and RAG context.
                        </p>
                        <select style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '1rem', outline: 'none' }}>
                            <option value="gpt2">GPT-2 (Local Fast Iterate)</option>
                            <option value="clinical-bert">Bio_ClinicalBERT</option>
                            <option value="llama-3">Llama-3-8B-Medical (Quantized)</option>
                        </select>
                    </div>

                    <div style={{ marginBottom: '2rem' }}>
                        <label style={{ display: 'block', fontWeight: 600, marginBottom: '8px' }}>Vector DB Top-K Parameter</label>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '12px' }}>
                            Number of related clinical guidelines retrieved to combat model hallucination.
                        </p>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <input type="range" min="1" max="10" defaultValue="3" style={{ flex: 1 }} />
                            <span style={{ background: '#f4f6fa', padding: '6px 16px', borderRadius: '6px', fontWeight: 600, border: '1px solid var(--border)' }}>3 Guidelines</span>
                        </div>
                    </div>

                    <div style={{ marginBottom: '2rem' }}>
                        <label style={{ display: 'block', fontWeight: 600, marginBottom: '8px' }}>Hardware Acceleration</label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer', padding: '12px', border: '1px solid var(--border)', borderRadius: '8px' }}>
                            <input type="checkbox" defaultChecked style={{ width: '18px', height: '18px', accentColor: 'var(--primary)' }} />
                            <div>
                                <div style={{ fontWeight: 500 }}>Use CUDA/MPS (GPU)</div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Accelerates 1D-CNN extraction and Transformer attention generation.</div>
                            </div>
                        </label>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', borderTop: '1px solid var(--border)', paddingTop: '1.5rem' }}>
                        <button className="outline-btn">Reset Defaults</button>
                        <button className="primary-btn">Save Changes</button>
                    </div>
                </div>

            </div>
        </div>
    );
};

export default Settings;
