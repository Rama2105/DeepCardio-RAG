import React from 'react';
import { Database, Server, Clock, Zap, Cpu, Network } from 'lucide-react';

const StatCard = ({ title, value, unit, icon: Icon, color, trend }) => (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h4 style={{ color: 'var(--text-muted)', fontSize: '0.9rem', fontWeight: 500 }}>{title}</h4>
            <div style={{ background: `${color}15`, padding: '8px', borderRadius: '8px', color: color }}>
                <Icon size={20} />
            </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-main)' }}>{value}</span>
            <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>{unit}</span>
        </div>
        {trend && (
            <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--success)', fontWeight: 500 }}>
                {trend}
            </div>
        )}
    </div>
);

const VectorDBStats = () => {
    return (
        <div className="view-container" style={{ padding: '2rem', height: '100%', overflowY: 'auto' }}>
            <div className="view-header" style={{ marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Database color="var(--primary)" />
                    Milvus Vector Database Health
                </h2>
                <p style={{ color: 'var(--text-muted)' }}>Real-time telemetry for the DeepCardio-RAG embedding retrieval system.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
                <StatCard title="Total Entities Indexed" value="50,142" unit="vectors" icon={Database} color="#0f62fe" trend="↑ 124 this week" />
                <StatCard title="Avg Retrieval Latency" value="14.2" unit="ms" icon={Zap} color="#fa4d56" trend="↓ 2.1ms improved" />
                <StatCard title="Memory Usage" value="1.2" unit="GB" icon={Cpu} color="#8a3ffc" trend="Stable pattern" />
                <StatCard title="Connection Status" value="Online" unit="" icon={Server} color="#198038" trend="Uptime: 99.99%" />
            </div>

            <div className="dashboard-grid" style={{ padding: 0, gap: '1.5rem' }}>
                <div className="card">
                    <div className="card-header">
                        <h3><Network size={20} color="var(--primary)" /> Collection Breakdown</h3>
                    </div>
                    <div style={{ padding: '1rem 0' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                            <span>Clinical Cases</span>
                            <strong>38,201 (76%)</strong>
                        </div>
                        <div style={{ width: '100%', height: '8px', background: 'var(--border)', borderRadius: '4px', marginBottom: '1.5rem' }}>
                            <div style={{ width: '76%', height: '100%', background: 'var(--primary)', borderRadius: '4px' }}></div>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                            <span>Medical Guidelines</span>
                            <strong>11,941 (24%)</strong>
                        </div>
                        <div style={{ width: '100%', height: '8px', background: 'var(--border)', borderRadius: '4px' }}>
                            <div style={{ width: '24%', height: '100%', background: '#8a3ffc', borderRadius: '4px' }}></div>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">
                        <h3><Clock size={20} color="var(--primary)" /> Embedder Statistics</h3>
                    </div>
                    <div style={{ padding: '1rem 0' }}>
                        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <li style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
                                <span style={{ color: 'var(--text-muted)' }}>Transformer Model:</span>
                                <strong style={{ fontFamily: 'monospace' }}>all-MiniLM-L6-v2</strong>
                            </li>
                            <li style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
                                <span style={{ color: 'var(--text-muted)' }}>Embedding Dimension:</span>
                                <strong>384-dimensional</strong>
                            </li>
                            <li style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
                                <span style={{ color: 'var(--text-muted)' }}>Distance Metric:</span>
                                <strong>COSINE</strong>
                            </li>
                            <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-muted)' }}>Index Type:</span>
                                <strong>FLAT</strong>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default VectorDBStats;
