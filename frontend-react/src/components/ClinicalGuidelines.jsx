import React from 'react';
import { BookOpen, ExternalLink, Bookmark } from 'lucide-react';

const GUIDELINES = [
    { id: 'GL-AHA-2023', title: '2023 AHA/ACC/ACCP/ASPC/NLA/PCNA Guideline for the Management of Patients With Chronic Coronary Disease', domain: 'Ischemia', date: 'Jul 2023' },
    { id: 'GL-ESC-AF', title: '2020 ESC Guidelines for the diagnosis and management of atrial fibrillation', domain: 'Arrhythmia', date: 'Aug 2020' },
    { id: 'GL-ACC-HF', title: '2022 AHA/ACC/HFSA Guideline for the Management of Heart Failure', domain: 'Heart Failure', date: 'Apr 2022' },
    { id: 'GL-ESC-VT', title: '2022 ESC Guidelines for the management of patients with ventricular arrhythmias', domain: 'Arrhythmia', date: 'Aug 2022' },
];

const ClinicalGuidelines = () => {
    return (
        <div className="view-container" style={{ padding: '2rem', height: '100%', overflowY: 'auto' }}>
            <div className="view-header" style={{ marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Clinical Guidelines Repository</h2>
                <p style={{ color: 'var(--text-muted)' }}>Browse the structured knowledge base actively used by the DeepCardio-RAG logic engine.</p>
            </div>

            <div style={{ display: 'grid', gap: '1.5rem', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))' }}>
                {GUIDELINES.map((gl, i) => (
                    <div key={i} className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                            <span className="db-badge">{gl.domain}</span>
                            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{gl.date}</span>
                        </div>
                        <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', lineHeight: 1.4, flex: 1 }}>{gl.title}</h3>

                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                            Reference ID: {gl.id}
                        </div>

                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button className="primary-btn" style={{ flex: 1, justifyContent: 'center', padding: '8px' }}>
                                <BookOpen size={16} /> Read Full Text
                            </button>
                            <button className="outline-btn" style={{ padding: '8px 12px' }}>
                                <Bookmark size={16} />
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ClinicalGuidelines;
