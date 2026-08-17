import React, { useState } from 'react';
import { Search, Filter, MoreVertical, FileText, Activity } from 'lucide-react';

const MOCK_PATIENTS = [
    { id: 'PT-8941', name: 'James Wilson', age: 62, status: 'Critical', lastECG: '10 mins ago', diagnosis: 'Pending AI Analysis' },
    { id: 'PT-8942', name: 'Sarah Chen', age: 45, status: 'Stable', lastECG: '2 hours ago', diagnosis: 'Normal Sinus Rhythm' },
    { id: 'PT-8943', name: 'Robert Fox', age: 71, status: 'Under Observation', lastECG: '1 day ago', diagnosis: 'Atrial Fibrillation' },
    { id: 'PT-8944', name: 'Maria Garcia', age: 58, status: 'Stable', lastECG: '1 day ago', diagnosis: 'Premature Ventricular Contractions' },
    { id: 'PT-8945', name: 'David Lee', age: 34, status: 'Discharged', lastECG: '3 days ago', diagnosis: 'Arrhythmogenic Right Ventricular Cardiomyopathy' },
];

const PatientRecords = () => {
    const [searchTerm, setSearchTerm] = useState('');

    const filteredPatients = MOCK_PATIENTS.filter(p =>
        p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.id.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="view-container" style={{ padding: '2rem', height: '100%', overflowY: 'auto' }}>
            <div className="view-header" style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Patient Records Directory</h2>
                    <p style={{ color: 'var(--text-muted)' }}>Manage and review historical patient ECG diagnostics across the ward.</p>
                </div>
                <button className="primary-btn">
                    <Activity size={18} /> New Patient
                </button>
            </div>

            <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
                <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border)', display: 'flex', gap: '1rem' }}>
                    <div className="search-bar" style={{ background: '#f4f6fa', flex: 1, maxWidth: '400px' }}>
                        <Search size={18} color="var(--text-muted)" />
                        <input
                            type="text"
                            placeholder="Search by name or ID..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                    <button className="outline-btn"><Filter size={18} /> Filter</button>
                </div>

                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                    <thead style={{ background: '#f9f9fb', borderBottom: '1px solid var(--border)', fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                        <tr>
                            <th style={{ padding: '1rem 1.5rem', fontWeight: 600 }}>Patient ID</th>
                            <th style={{ padding: '1rem 1.5rem', fontWeight: 600 }}>Name & Age</th>
                            <th style={{ padding: '1rem 1.5rem', fontWeight: 600 }}>Status</th>
                            <th style={{ padding: '1rem 1.5rem', fontWeight: 600 }}>Last ECG</th>
                            <th style={{ padding: '1rem 1.5rem', fontWeight: 600 }}>AI Diagnosis Focus</th>
                            <th style={{ padding: '1rem 1.5rem', fontWeight: 600 }}>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredPatients.map((patient, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.2s' }} className="table-row-hover">
                                <td style={{ padding: '1rem 1.5rem', fontWeight: 500, color: 'var(--primary)' }}>{patient.id}</td>
                                <td style={{ padding: '1rem 1.5rem' }}>
                                    <div style={{ fontWeight: 500 }}>{patient.name}</div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{patient.age} yrs</div>
                                </td>
                                <td style={{ padding: '1rem 1.5rem' }}>
                                    <span style={{
                                        padding: '4px 10px',
                                        borderRadius: '20px',
                                        fontSize: '0.75rem',
                                        fontWeight: 600,
                                        background: patient.status === 'Critical' ? '#ffe1e1' : patient.status === 'Stable' ? '#e1ffe1' : '#f0f4ff',
                                        color: patient.status === 'Critical' ? '#d90000' : patient.status === 'Stable' ? '#008a00' : 'var(--primary)'
                                    }}>
                                        {patient.status}
                                    </span>
                                </td>
                                <td style={{ padding: '1rem 1.5rem', fontSize: '0.9rem', color: 'var(--text-muted)' }}>{patient.lastECG}</td>
                                <td style={{ padding: '1rem 1.5rem', fontSize: '0.9rem' }}>{patient.diagnosis}</td>
                                <td style={{ padding: '1rem 1.5rem' }}>
                                    <div style={{ display: 'flex', gap: '8px' }}>
                                        <button className="icon-btn" title="View Report"><FileText size={16} /></button>
                                        <button className="icon-btn" title="More Options"><MoreVertical size={16} /></button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>

                {filteredPatients.length === 0 && (
                    <div className="empty-state">
                        <p>No patients found matching your search.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PatientRecords;
