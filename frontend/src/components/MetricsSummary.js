import React, { useEffect, useState } from 'react';

export default function MetricsSummary() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch("https://graph-neural-networks-for-survival.onrender.com/metrics")
      .then(res => res.json())
      .then(data => setMetrics(data));
  }, []);

  if (!metrics) return <p>Loading...</p>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold">F1 Score: {metrics.f1_score}</h2>
    </div>
  );
}