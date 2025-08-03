import { Line } from "react-chartjs-2";
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement, Legend } from "chart.js";
ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend);

export default function Chart({ trainLoss, valLoss }) {
  const data = {
    labels: Array.from({ length: trainLoss.length }, (_, i) => i + 1),
    datasets: [
      {
        label: 'Train Loss',
        data: trainLoss,
        borderColor: 'rgb(59,130,246)',
      },
      {
        label: 'Val Loss',
        data: valLoss,
        borderColor: 'rgb(234,88,12)',
      }
    ]
  };

  return <Line data={data} />;
}