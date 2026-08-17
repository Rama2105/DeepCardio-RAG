import React, { useEffect, useState, useRef } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const ECGChart = ({ isAnalyzing }) => {
    const [data, setData] = useState(Array(100).fill(0));
    const timeRef = useRef(0);
    const reqRef = useRef(null);

    const chartData = {
        labels: Array.from({ length: 100 }, (_, i) => i),
        datasets: [
            {
                label: 'Lead II (Simulated)',
                borderColor: '#0f62fe',
                borderWidth: 2,
                data: data,
                pointRadius: 0,
                tension: 0.4
            }
        ]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
            x: {
                display: true,
                grid: { color: 'rgba(255, 255, 255, 0.1)', lineWidth: 1 },
                ticks: { display: false }
            },
            y: {
                display: true,
                min: -2,
                max: 3,
                grid: { color: 'rgba(255, 255, 255, 0.1)', lineWidth: 1 },
                ticks: { display: false }
            }
        },
        plugins: {
            legend: { display: true, labels: { color: '#fff' } }
        }
    };

    useEffect(() => {
        const tick = () => {
            if (isAnalyzing) return; // Freeze frame when backend analyzing

            setData(prev => {
                const newData = [...prev.slice(1)];
                const cycle = timeRef.current % 100;
                let val = 0;

                if (cycle === 10) val = 0.5;
                else if (cycle === 12) val = -0.3;
                else if (cycle === 14) val = 2.5; // Spike
                else if (cycle === 16) val = -0.5;
                else if (cycle === 30) val = 0.7;
                else val = (Math.random() * 0.1) - 0.05;

                newData.push(val);
                timeRef.current++;
                return newData;
            });
            reqRef.current = requestAnimationFrame(tick);
        };

        reqRef.current = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(reqRef.current);
    }, [isAnalyzing]);

    return <Line data={chartData} options={options} />;
};

export default ECGChart;
